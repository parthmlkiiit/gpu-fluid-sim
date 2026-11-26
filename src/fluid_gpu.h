// Copyright 2026 Parth
// Stable-fluids solver on the GPU: owns the device fields and runs the
// NVRTC-compiled kernels in the order of one simulation step.

#ifndef SRC_FLUID_GPU_H_
#define SRC_FLUID_GPU_H_

#include <cuda.h>

#include <vector>

#include "src/config.h"
#include "src/cuda_context.h"
#include "src/fluid_common.h"

namespace fluid {

class GpuFluid {
 public:
  GpuFluid(const Config& cfg, const KernelModule& module);

  GpuFluid(const GpuFluid&) = delete;
  GpuFluid& operator=(const GpuFluid&) = delete;

  // True if every buffer was allocated and every kernel was found.
  bool ok() const { return ok_; }

  // Advances the simulation by one step and reports GPU time per phase.
  bool Step(int step, StepTimes* times);

  // Dye -> RGB8 on the GPU, then copies the n*n*3 bytes back.
  bool Render(std::vector<unsigned char>* rgb);

  // RMS divergence of the current velocity (0 = perfectly incompressible).
  bool ResidualDivergence(double* rms);

  bool Download(HostFields* fields) const;

 private:
  Config cfg_;
  int n_;
  bool ok_ = false;

  DeviceArray vx_, vy_, vx_next_, vy_next_;
  DeviceArray pressure_, pressure_next_, divergence_, curl_;
  DeviceArray red_, green_, blue_, red_next_, green_next_, blue_next_;
  DeviceArray rgb_;

  CUfunction apply_emitters_ = nullptr;
  CUfunction advect_ = nullptr;
  CUfunction curl_kernel_ = nullptr;
  CUfunction vorticity_ = nullptr;
  CUfunction divergence_kernel_ = nullptr;
  CUfunction jacobi_ = nullptr;
  CUfunction subtract_gradient_ = nullptr;
  CUfunction set_boundary_ = nullptr;
  CUfunction render_ = nullptr;

  EventTimeline timeline_;
};

}  // namespace fluid

#endif  // SRC_FLUID_GPU_H_
