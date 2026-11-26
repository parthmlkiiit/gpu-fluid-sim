// Copyright 2026 Parth
// Run settings for the fluid simulation and the command line parser.

#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#include <string>

namespace fluid {

struct Config {
  int grid = 512;  // simulation is grid x grid cells
  int steps = 600;
  float dt = 1.0f;                      // time step, in cells per unit velocity
  int jacobi_iters = 40;                // pressure solver sweeps per step
  std::string jacobi_kernel = "tiled";  // "tiled" (shared memory) or "naive"
  float vorticity = 0.3f;  // vorticity confinement strength, 0 = off
  float velocity_dissipation = 0.999f;
  float dye_dissipation = 0.997f;
  float force = 1.2f;         // emitter push, for a 512 grid
  int frame_every = 0;        // save a PNG every N steps, 0 = never
  int log_every = 50;         // progress line + divergence check
  int cpu_compare_steps = 0;  // also run the CPU solver for N steps
  std::string output_dir = "output";
  std::string kernel_file = "src/kernels/fluid.cu";
  std::string summary_csv;  // append one summary row here if set
  int device = 0;
  bool show_help = false;
};

// Fills cfg from argv (flags look like --name=value). Prints the reason and
// returns false on a bad flag or value.
bool ParseArgs(int argc, char** argv, Config* cfg);

void PrintUsage(const char* program);

void PrintConfig(const Config& cfg);

}  // namespace fluid

#endif  // SRC_CONFIG_H_
