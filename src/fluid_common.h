// Copyright 2026 Parth
// Things the GPU and CPU solvers share: the emitter script, the host copy of
// the fields and the per-step timings.

#ifndef SRC_FLUID_COMMON_H_
#define SRC_FLUID_COMMON_H_

#include <vector>

namespace fluid {

constexpr int kMaxEmitters = 4;

// Must match struct EmitterSet in src/kernels/fluid.cu (passed by value to
// the ApplyEmitters kernel, so the layout has to be identical).
struct EmitterSet {
  int count;
  float radius;
  float x[kMaxEmitters];
  float y[kMaxEmitters];
  float fx[kMaxEmitters];
  float fy[kMaxEmitters];
  float r[kMaxEmitters];
  float g[kMaxEmitters];
  float b[kMaxEmitters];
};

// Four jets on a slowly turning ring, aimed slightly off-centre so they wind
// into a vortex, pulsing on and off. Depends only on (step, n, force), so the
// GPU and CPU runs get exactly the same input.
EmitterSet MakeEmitters(int step, int n, float force);

// Host copy of the fields that matter for comparing two solvers.
struct HostFields {
  std::vector<float> vx;
  std::vector<float> vy;
  std::vector<float> red;
  std::vector<float> green;
  std::vector<float> blue;
};

// GPU time of each phase of one step, from CUDA events.
struct StepTimes {
  float forces_ms = 0.0f;    // emitters + vorticity confinement
  float advect_ms = 0.0f;    // velocity self-advection
  float pressure_ms = 0.0f;  // divergence + Jacobi + gradient + walls
  float dye_ms = 0.0f;       // dye advection
  float total_ms = 0.0f;
};

}  // namespace fluid

#endif  // SRC_FLUID_COMMON_H_
