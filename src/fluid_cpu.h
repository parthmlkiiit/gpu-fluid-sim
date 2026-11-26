// Copyright 2026 Parth
// Single-threaded CPU version of the same solver, used to check the GPU
// results and to measure the speedup. Mirrors src/kernels/fluid.cu.

#ifndef SRC_FLUID_CPU_H_
#define SRC_FLUID_CPU_H_

#include <vector>

#include "src/config.h"
#include "src/fluid_common.h"

namespace fluid {

class CpuFluid {
 public:
  explicit CpuFluid(const Config& cfg);

  void Step(int step);

  const HostFields& fields() const { return fields_; }

 private:
  Config cfg_;
  int n_;
  HostFields fields_;
  std::vector<float> vx_next_, vy_next_, pressure_, pressure_next_;
  std::vector<float> divergence_, curl_, dye_next_;
};

}  // namespace fluid

#endif  // SRC_FLUID_CPU_H_
