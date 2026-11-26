// Copyright 2026 Parth
// gpu_fluid: 2-D smoke / dye simulation (Stam's stable fluids) with custom
// CUDA kernels compiled at runtime by NVRTC.
//
//   ./bin/gpu_fluid --grid=1024 --steps=900 --frame_every=2
//
// See README.md for every flag.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "src/config.h"
#include "src/cuda_context.h"
#include "src/fluid_common.h"
#include "src/fluid_cpu.h"
#include "src/fluid_gpu.h"
#include "src/image_writer.h"

namespace fluid {
namespace {

using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct RunSummary {
  double first_step_ms = 0.0;
  double avg_step_ms = 0.0;  // all steps except the first
  double avg_forces_ms = 0.0;
  double avg_advect_ms = 0.0;
  double avg_pressure_ms = 0.0;
  double avg_dye_ms = 0.0;
  double final_div_rms = 0.0;
  double wall_ms = 0.0;
  int frames = 0;
  int writer_threads = 0;
  // Filled only when --cpu_compare_steps > 0.
  double cpu_step_ms = 0.0;
  double max_velocity_diff = 0.0;
  double max_dye_diff = 0.0;
  double max_velocity = 0.0;
  double max_dye = 0.0;
};

bool RunSimulation(const Config& cfg, const KernelModule& module,
                   RunSummary* summary) {
  GpuFluid sim(cfg, module);
  if (!sim.ok()) return false;

  const std::string frame_dir = cfg.output_dir + "/frames";
  if (!MakeDirs(cfg.output_dir) ||
      (cfg.frame_every > 0 && !MakeDirs(frame_dir))) {
    return false;
  }
  const std::string csv_path = cfg.output_dir + "/steps.csv";
  std::FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (csv == nullptr) {
    std::printf("error: cannot open %s\n", csv_path.c_str());
    return false;
  }
  std::fprintf(csv,
               "step,total_ms,forces_ms,advect_ms,pressure_ms,dye_ms,"
               "div_rms\n");

  // PNG encoding is slow (~100+ ms for 1024x1024), so it runs on spare CPU
  // cores; otherwise the GPU idles between steps and the laptop drops clocks.
  const int cores = static_cast<int>(std::thread::hardware_concurrency());
  const int writer_threads =
      cfg.frame_every > 0 ? std::clamp(cores - 1, 1, 8) : 0;
  FrameWriter writer(writer_threads, 16);
  summary->writer_threads = writer_threads;

  std::printf("running %d steps\n\n", cfg.steps);
  std::vector<unsigned char> rgb;
  StepTimes sum;
  const Clock::time_point start = Clock::now();
  bool ok = true;

  for (int step = 0; step < cfg.steps && ok; ++step) {
    StepTimes t;
    ok = sim.Step(step, &t);
    if (!ok) break;
    if (step == 0) {
      summary->first_step_ms = t.total_ms;
    } else {
      sum.forces_ms += t.forces_ms;
      sum.advect_ms += t.advect_ms;
      sum.pressure_ms += t.pressure_ms;
      sum.dye_ms += t.dye_ms;
      sum.total_ms += t.total_ms;
    }

    double div_rms = -1.0;
    const bool log = (step + 1) % cfg.log_every == 0 || step == cfg.steps - 1;
    if (log) {
      ok = sim.ResidualDivergence(&div_rms);
      summary->final_div_rms = div_rms;
    }
    if (ok && cfg.frame_every > 0 && step % cfg.frame_every == 0) {
      char name[64];
      std::snprintf(name, sizeof(name), "/frame_%05d.png", summary->frames);
      ok = sim.Render(&rgb);
      if (ok)
        writer.Write(frame_dir + name, cfg.grid, cfg.grid, std::move(rgb));
      ++summary->frames;
    }

    std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,", step, t.total_ms,
                 t.forces_ms, t.advect_ms, t.pressure_ms, t.dye_ms);
    if (div_rms >= 0.0) std::fprintf(csv, "%.6e", div_rms);
    std::fprintf(csv, "\n");

    if (log) {
      std::printf(
          "[step %5d/%d] %8.3f ms  (forces %.3f  advect %.3f  pressure %.3f"
          "  dye %.3f)  div_rms %.2e\n",
          step + 1, cfg.steps, t.total_ms, t.forces_ms, t.advect_ms,
          t.pressure_ms, t.dye_ms, div_rms);
    }
  }
  std::fclose(csv);
  ok = writer.Finish() && ok;
  summary->wall_ms = MsSince(start);

  const int timed = std::max(1, cfg.steps - 1);
  summary->avg_step_ms = sum.total_ms / timed;
  summary->avg_forces_ms = sum.forces_ms / timed;
  summary->avg_advect_ms = sum.advect_ms / timed;
  summary->avg_pressure_ms = sum.pressure_ms / timed;
  summary->avg_dye_ms = sum.dye_ms / timed;
  return ok;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b,
                  double* max_value) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    *max_value = std::max(*max_value, static_cast<double>(std::fabs(b[i])));
  }
  return worst;
}

// Runs a fresh GPU and CPU solver from the same start for
// cfg.cpu_compare_steps steps and compares the fields. The GPU steps run
// back to back first: interleaving them with multi-second CPU steps lets the
// GPU clock down and makes it look slower than it is. Speedup is taken
// against the steady-state GPU average of the main run.
bool CompareWithCpu(const Config& cfg, const KernelModule& module,
                    RunSummary* summary) {
  std::printf("\ncomparing with the CPU solver for %d steps at %d x %d\n",
              cfg.cpu_compare_steps, cfg.grid, cfg.grid);
  GpuFluid gpu(cfg, module);
  if (!gpu.ok()) return false;
  for (int step = 0; step < cfg.cpu_compare_steps; ++step) {
    StepTimes t;
    if (!gpu.Step(step, &t)) return false;
  }

  CpuFluid cpu(cfg);
  const Clock::time_point start = Clock::now();
  for (int step = 0; step < cfg.cpu_compare_steps; ++step) cpu.Step(step);
  const double cpu_ms = MsSince(start);

  HostFields g;
  if (!gpu.Download(&g)) return false;
  const HostFields& c = cpu.fields();
  summary->max_velocity_diff =
      std::max(MaxAbsDiff(g.vx, c.vx, &summary->max_velocity),
               MaxAbsDiff(g.vy, c.vy, &summary->max_velocity));
  summary->max_dye_diff =
      std::max({MaxAbsDiff(g.red, c.red, &summary->max_dye),
                MaxAbsDiff(g.green, c.green, &summary->max_dye),
                MaxAbsDiff(g.blue, c.blue, &summary->max_dye)});
  summary->cpu_step_ms = cpu_ms / cfg.cpu_compare_steps;
  return true;
}

void PrintSummary(const Config& cfg, const RunSummary& s) {
  const double cells = cfg.grid * static_cast<double>(cfg.grid);
  const double steps_per_s = s.avg_step_ms > 0.0 ? 1000.0 / s.avg_step_ms : 0;
  auto pct = [&s](double part) {
    return s.avg_step_ms > 0.0 ? 100.0 * part / s.avg_step_ms : 0.0;
  };
  std::printf("\n================ summary ================\n");
  std::printf("grid               : %d x %d\n", cfg.grid, cfg.grid);
  std::printf("steps / frames     : %d / %d\n", cfg.steps, s.frames);
  std::printf("GPU time per step  : %.3f ms avg (first step %.3f ms)\n",
              s.avg_step_ms, s.first_step_ms);
  std::printf("  forces           : %6.3f ms (%4.1f%%)\n", s.avg_forces_ms,
              pct(s.avg_forces_ms));
  std::printf("  advect velocity  : %6.3f ms (%4.1f%%)\n", s.avg_advect_ms,
              pct(s.avg_advect_ms));
  std::printf("  pressure solve   : %6.3f ms (%4.1f%%)  %d x %s Jacobi\n",
              s.avg_pressure_ms, pct(s.avg_pressure_ms), cfg.jacobi_iters,
              cfg.jacobi_kernel.c_str());
  std::printf("  advect dye       : %6.3f ms (%4.1f%%)\n", s.avg_dye_ms,
              pct(s.avg_dye_ms));
  std::printf("throughput         : %.1f steps/s, %.1f M cell-steps/s\n",
              steps_per_s, steps_per_s * cells / 1e6);
  std::printf("residual div (rms) : %.3e\n", s.final_div_rms);
  std::printf("wall time          : %.2f s", s.wall_ms / 1000.0);
  if (s.writer_threads > 0) {
    std::printf(" (PNGs written by %d background threads)", s.writer_threads);
  }
  std::printf("\n");
  if (cfg.cpu_compare_steps > 0) {
    std::printf("CPU step (1 core)  : %.2f ms\n", s.cpu_step_ms);
    std::printf("GPU step (avg)     : %.3f ms  -> speedup %.1fx\n",
                s.avg_step_ms, s.cpu_step_ms / s.avg_step_ms);
    std::printf(
        "max |GPU - CPU|    : velocity %.2e (max |v| %.3f), "
        "dye %.2e (max %.3f)\n",
        s.max_velocity_diff, s.max_velocity, s.max_dye_diff, s.max_dye);
  }
}

bool AppendSummaryCsv(const Config& cfg, const RunSummary& s) {
  std::FILE* probe = std::fopen(cfg.summary_csv.c_str(), "r");
  const bool exists = probe != nullptr;
  if (probe != nullptr) std::fclose(probe);

  std::FILE* csv = std::fopen(cfg.summary_csv.c_str(), "a");
  if (csv == nullptr) {
    std::printf("error: cannot open %s\n", cfg.summary_csv.c_str());
    return false;
  }
  if (!exists) {
    std::fprintf(csv,
                 "grid,steps,jacobi_iters,jacobi_kernel,avg_step_ms,"
                 "pressure_ms,steps_per_s,mcell_steps_per_s,div_rms,"
                 "cpu_step_ms,speedup,max_velocity_diff,max_dye_diff\n");
  }
  const double steps_per_s = 1000.0 / s.avg_step_ms;
  const double speedup = s.cpu_step_ms / s.avg_step_ms;
  std::fprintf(csv,
               "%d,%d,%d,%s,%.4f,%.4f,%.2f,%.2f,%.4e,%.3f,%.2f,%.3e,%.3e\n",
               cfg.grid, cfg.steps, cfg.jacobi_iters, cfg.jacobi_kernel.c_str(),
               s.avg_step_ms, s.avg_pressure_ms, steps_per_s,
               steps_per_s * cfg.grid * static_cast<double>(cfg.grid) / 1e6,
               s.final_div_rms, s.cpu_step_ms, speedup, s.max_velocity_diff,
               s.max_dye_diff);
  std::fclose(csv);
  return true;
}

}  // namespace
}  // namespace fluid

int main(int argc, char** argv) {
  fluid::Config cfg;
  if (!fluid::ParseArgs(argc, argv, &cfg)) {
    std::printf("\n");
    fluid::PrintUsage(argv[0]);
    return 1;
  }
  if (cfg.show_help) {
    fluid::PrintUsage(argv[0]);
    return 0;
  }

  std::printf("==== gpu_fluid: stable fluids with NVRTC-compiled CUDA ====\n");
  fluid::CudaContext context;
  if (!context.Init(cfg.device)) return 1;
  fluid::KernelModule module;
  if (!module.Build(cfg.kernel_file, context.cc_major(), context.cc_minor())) {
    return 1;
  }
  fluid::PrintConfig(cfg);

  fluid::RunSummary summary;
  if (!fluid::RunSimulation(cfg, module, &summary)) return 2;
  if (cfg.cpu_compare_steps > 0 &&
      !fluid::CompareWithCpu(cfg, module, &summary)) {
    return 2;
  }
  fluid::PrintSummary(cfg, summary);
  if (!cfg.summary_csv.empty() && !fluid::AppendSummaryCsv(cfg, summary)) {
    return 2;
  }
  return 0;
}
