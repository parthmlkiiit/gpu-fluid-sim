// Copyright 2026 Parth
#include "src/fluid_gpu.h"

#include <cuda.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "src/config.h"
#include "src/cuda_context.h"
#include "src/fluid_common.h"

namespace fluid {
namespace {

size_t FieldBytes(int n) { return static_cast<size_t>(n) * n * sizeof(float); }

// Phase boundaries recorded inside Step().
enum Mark { kStart, kForcesDone, kAdvectDone, kPressureDone, kDyeDone, kMarks };

}  // namespace

GpuFluid::GpuFluid(const Config& cfg, const KernelModule& module)
    : cfg_(cfg),
      n_(cfg.grid),
      vx_(FieldBytes(n_)),
      vy_(FieldBytes(n_)),
      vx_next_(FieldBytes(n_)),
      vy_next_(FieldBytes(n_)),
      pressure_(FieldBytes(n_)),
      pressure_next_(FieldBytes(n_)),
      divergence_(FieldBytes(n_)),
      curl_(FieldBytes(n_)),
      red_(FieldBytes(n_)),
      green_(FieldBytes(n_)),
      blue_(FieldBytes(n_)),
      red_next_(FieldBytes(n_)),
      green_next_(FieldBytes(n_)),
      blue_next_(FieldBytes(n_)),
      rgb_(static_cast<size_t>(n_) * n_ * 3),
      timeline_(kMarks) {
  apply_emitters_ = module.Get("ApplyEmitters");
  advect_ = module.Get("Advect");
  curl_kernel_ = module.Get("Curl");
  vorticity_ = module.Get("Vorticity");
  divergence_kernel_ = module.Get("Divergence");
  jacobi_ =
      module.Get(cfg.jacobi_kernel == "naive" ? "JacobiNaive" : "JacobiTiled");
  subtract_gradient_ = module.Get("SubtractGradient");
  set_boundary_ = module.Get("SetBoundary");
  render_ = module.Get("Render");

  const DeviceArray* fields[] = {&vx_,         &vy_,        &vx_next_,
                                 &vy_next_,    &pressure_,  &pressure_next_,
                                 &divergence_, &curl_,      &red_,
                                 &green_,      &blue_,      &red_next_,
                                 &green_next_, &blue_next_, &rgb_};
  ok_ = apply_emitters_ && advect_ && curl_kernel_ && vorticity_ &&
        divergence_kernel_ && jacobi_ && subtract_gradient_ && set_boundary_ &&
        render_;
  for (const DeviceArray* field : fields) ok_ = ok_ && field->ok();
  if (!ok_) return;

  // Fluid at rest, no dye, zero pressure.
  for (DeviceArray* field : {&vx_, &vy_, &pressure_, &red_, &green_, &blue_}) {
    ok_ = ok_ && field->Zero();
  }
}

bool GpuFluid::Step(int step, StepTimes* times) {
  const int n = n_;
  const float dt = cfg_.dt;
  const EmitterSet emitters = MakeEmitters(step, n, cfg_.force);

  bool ok = timeline_.Mark(kStart);

  // 1. Forces: emitters push the fluid and inject dye, then vorticity
  //    confinement re-adds the small swirls that advection smooths away.
  ok = ok && Launch2D(apply_emitters_, n, vx_.get(), vy_.get(), red_.get(),
                      green_.get(), blue_.get(), n, emitters);
  if (cfg_.vorticity > 0.0f) {
    ok = ok && Launch2D(curl_kernel_, n, vx_.get(), vy_.get(), curl_.get(), n);
    ok = ok && Launch2D(vorticity_, n, vx_.get(), vy_.get(), curl_.get(), n, dt,
                        cfg_.vorticity);
  }
  ok = ok && timeline_.Mark(kForcesDone);

  // 2. The velocity field carries itself along (both components read the
  //    old field, so they go to the *_next buffers first).
  ok = ok && Launch2D(advect_, n, vx_.get(), vx_next_.get(), vx_.get(),
                      vy_.get(), n, dt, cfg_.velocity_dissipation);
  ok = ok && Launch2D(advect_, n, vy_.get(), vy_next_.get(), vx_.get(),
                      vy_.get(), n, dt, cfg_.velocity_dissipation);
  vx_.Swap(&vx_next_);
  vy_.Swap(&vy_next_);
  ok = ok && timeline_.Mark(kAdvectDone);

  // 3. Projection: solve laplace(p) = div(v) with Jacobi sweeps (warm-started
  //    from last step's pressure) and subtract grad(p) so div(v) ~ 0.
  ok = ok && Launch2D(divergence_kernel_, n, vx_.get(), vy_.get(),
                      divergence_.get(), n);
  for (int i = 0; i < cfg_.jacobi_iters && ok; ++i) {
    ok = Launch2D(jacobi_, n, pressure_.get(), pressure_next_.get(),
                  divergence_.get(), n);
    pressure_.Swap(&pressure_next_);
  }
  ok = ok && Launch2D(subtract_gradient_, n, vx_.get(), vy_.get(),
                      pressure_.get(), n);
  ok = ok && Launch2D(set_boundary_, n, vx_.get(), vy_.get(), n);
  ok = ok && timeline_.Mark(kPressureDone);

  // 4. Dye is carried by the now divergence-free velocity.
  DeviceArray* dye[] = {&red_, &green_, &blue_};
  DeviceArray* dye_next[] = {&red_next_, &green_next_, &blue_next_};
  for (int c = 0; c < 3; ++c) {
    ok = ok && Launch2D(advect_, n, dye[c]->get(), dye_next[c]->get(),
                        vx_.get(), vy_.get(), n, dt, cfg_.dye_dissipation);
    dye[c]->Swap(dye_next[c]);
  }
  ok = ok && timeline_.Mark(kDyeDone) && timeline_.Sync();
  if (!ok) return false;

  times->forces_ms = timeline_.Ms(kStart, kForcesDone);
  times->advect_ms = timeline_.Ms(kForcesDone, kAdvectDone);
  times->pressure_ms = timeline_.Ms(kAdvectDone, kPressureDone);
  times->dye_ms = timeline_.Ms(kPressureDone, kDyeDone);
  times->total_ms = timeline_.Ms(kStart, kDyeDone);
  return true;
}

bool GpuFluid::Render(std::vector<unsigned char>* rgb) {
  if (!Launch2D(render_, n_, red_.get(), green_.get(), blue_.get(), rgb_.get(),
                n_)) {
    return false;
  }
  rgb->resize(rgb_.bytes());
  return rgb_.Download(rgb->data());
}

bool GpuFluid::ResidualDivergence(double* rms) {
  if (!Launch2D(divergence_kernel_, n_, vx_.get(), vy_.get(), divergence_.get(),
                n_)) {
    return false;
  }
  std::vector<float> div(static_cast<size_t>(n_) * n_);
  if (!divergence_.Download(div.data())) return false;
  double sum = 0.0;
  for (float d : div) sum += static_cast<double>(d) * d;
  *rms = std::sqrt(sum / div.size());
  return true;
}

bool GpuFluid::Download(HostFields* fields) const {
  const size_t cells = static_cast<size_t>(n_) * n_;
  fields->vx.resize(cells);
  fields->vy.resize(cells);
  fields->red.resize(cells);
  fields->green.resize(cells);
  fields->blue.resize(cells);
  return vx_.Download(fields->vx.data()) && vy_.Download(fields->vy.data()) &&
         red_.Download(fields->red.data()) &&
         green_.Download(fields->green.data()) &&
         blue_.Download(fields->blue.data());
}

}  // namespace fluid
