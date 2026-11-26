// Copyright 2026 Parth
#include "src/config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace fluid {
namespace {

// Rejects things like "5x" or "" instead of silently reading 5 or 0.
bool ParseInt(const std::string& text, int* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const int64_t parsed = std::strtoll(text.c_str(), &end, 10);
  if (*end != '\0' || parsed < INT32_MIN || parsed > INT32_MAX) return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseFloat(const std::string& text, float* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (*end != '\0' || !std::isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

bool Fail(const char* message) {
  std::printf("error: %s\n", message);
  return false;
}

bool Validate(const Config& cfg) {
  if (cfg.grid < 32 || cfg.grid > 4096) {
    return Fail("--grid must be in [32, 4096]");
  }
  if (cfg.steps < 1) return Fail("--steps must be >= 1");
  if (cfg.dt <= 0.0f || cfg.dt > 10.0f) return Fail("--dt must be in (0, 10]");
  if (cfg.jacobi_iters < 1 || cfg.jacobi_iters > 10000) {
    return Fail("--jacobi_iters must be in [1, 10000]");
  }
  if (cfg.jacobi_kernel != "tiled" && cfg.jacobi_kernel != "naive") {
    return Fail("--jacobi_kernel must be tiled or naive");
  }
  if (cfg.vorticity < 0.0f) return Fail("--vorticity must be >= 0");
  if (cfg.velocity_dissipation <= 0.0f || cfg.velocity_dissipation > 1.0f ||
      cfg.dye_dissipation <= 0.0f || cfg.dye_dissipation > 1.0f) {
    return Fail("dissipation values must be in (0, 1]");
  }
  if (cfg.frame_every < 0 || cfg.log_every < 1 || cfg.cpu_compare_steps < 0 ||
      cfg.device < 0) {
    return Fail(
        "--frame_every, --cpu_compare_steps, --device must be >= 0 "
        "and --log_every >= 1");
  }
  return true;
}

}  // namespace

bool ParseArgs(int argc, char** argv, Config* cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      cfg->show_help = true;
      return true;
    }
    const size_t eq = arg.find('=');
    if (arg.rfind("--", 0) != 0 || eq == std::string::npos) {
      std::printf("error: bad argument '%s' (expected --name=value)\n",
                  arg.c_str());
      return false;
    }
    const std::string key = arg.substr(2, eq - 2);
    const std::string value = arg.substr(eq + 1);

    bool ok = true;
    if (key == "grid") {
      ok = ParseInt(value, &cfg->grid);
    } else if (key == "steps") {
      ok = ParseInt(value, &cfg->steps);
    } else if (key == "dt") {
      ok = ParseFloat(value, &cfg->dt);
    } else if (key == "jacobi_iters") {
      ok = ParseInt(value, &cfg->jacobi_iters);
    } else if (key == "jacobi_kernel") {
      cfg->jacobi_kernel = value;
    } else if (key == "vorticity") {
      ok = ParseFloat(value, &cfg->vorticity);
    } else if (key == "velocity_dissipation") {
      ok = ParseFloat(value, &cfg->velocity_dissipation);
    } else if (key == "dye_dissipation") {
      ok = ParseFloat(value, &cfg->dye_dissipation);
    } else if (key == "force") {
      ok = ParseFloat(value, &cfg->force);
    } else if (key == "frame_every") {
      ok = ParseInt(value, &cfg->frame_every);
    } else if (key == "log_every") {
      ok = ParseInt(value, &cfg->log_every);
    } else if (key == "cpu_compare_steps") {
      ok = ParseInt(value, &cfg->cpu_compare_steps);
    } else if (key == "output_dir") {
      cfg->output_dir = value;
    } else if (key == "kernel_file") {
      cfg->kernel_file = value;
    } else if (key == "summary_csv") {
      cfg->summary_csv = value;
    } else if (key == "device") {
      ok = ParseInt(value, &cfg->device);
    } else {
      std::printf("error: unknown flag --%s\n", key.c_str());
      return false;
    }
    if (!ok) {
      std::printf("error: bad value for --%s: '%s'\n", key.c_str(),
                  value.c_str());
      return false;
    }
  }
  return Validate(*cfg);
}

void PrintUsage(const char* program) {
  std::printf(
      "usage: %s [--name=value ...]\n"
      "\n"
      "simulation\n"
      "  --grid=N                  grid size N x N, 32..4096 (default 512)\n"
      "  --steps=N                 time steps to run (default 600)\n"
      "  --dt=F                    time step (default 1.0)\n"
      "  --jacobi_iters=N          pressure solver sweeps per step "
      "(default 40)\n"
      "  --jacobi_kernel=K         tiled (shared memory) or naive "
      "(default tiled)\n"
      "  --vorticity=F             vorticity confinement, 0 = off "
      "(default 0.3)\n"
      "  --velocity_dissipation=F  per-step velocity decay (default 0.999)\n"
      "  --dye_dissipation=F       per-step dye decay (default 0.997)\n"
      "  --force=F                 emitter strength (default 1.2)\n"
      "\n"
      "output\n"
      "  --output_dir=DIR          frames/, steps.csv go here "
      "(default output)\n"
      "  --frame_every=N           save a PNG every N steps, 0 = none "
      "(default 0)\n"
      "  --log_every=N             progress + divergence every N steps "
      "(default 50)\n"
      "  --cpu_compare_steps=N     also run the CPU solver N steps, compare "
      "fields and speed (default 0)\n"
      "  --summary_csv=FILE        append a one-line summary to FILE\n"
      "\n"
      "other\n"
      "  --kernel_file=FILE        CUDA source compiled at runtime "
      "(default src/kernels/fluid.cu)\n"
      "  --device=N                CUDA device id (default 0)\n"
      "  --help                    show this message\n",
      program);
}

void PrintConfig(const Config& cfg) {
  std::printf("grid           : %d x %d (%.2f M cells)\n", cfg.grid, cfg.grid,
              cfg.grid * static_cast<double>(cfg.grid) / 1e6);
  std::printf("steps / dt     : %d / %.2f\n", cfg.steps, cfg.dt);
  std::printf("pressure       : %d Jacobi sweeps, %s kernel\n",
              cfg.jacobi_iters, cfg.jacobi_kernel.c_str());
  std::printf("vorticity      : %.2f\n", cfg.vorticity);
  std::printf("dissipation    : velocity %.3f, dye %.3f\n",
              cfg.velocity_dissipation, cfg.dye_dissipation);
  std::printf(
      "frames         : %s\n",
      cfg.frame_every > 0
          ? ("every " + std::to_string(cfg.frame_every) + " steps").c_str()
          : "off");
  std::printf("cpu compare    : %d steps\n\n", cfg.cpu_compare_steps);
}

}  // namespace fluid
