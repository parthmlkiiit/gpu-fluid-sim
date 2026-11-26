// Copyright 2026 Parth
// Stable fluids (Stam 1999) on a square grid, compiled at runtime with NVRTC.
// Every field is an n*n float array stored row by row, cell size 1.
// Launch layout for every kernel: 16x16 blocks covering the n x n grid.

#define TILE 16

// Must match fluid::EmitterSet in src/emitters.h (passed by value).
#define MAX_EMITTERS 4
struct EmitterSet {
  int count;
  float radius;
  float x[MAX_EMITTERS];
  float y[MAX_EMITTERS];
  float fx[MAX_EMITTERS];
  float fy[MAX_EMITTERS];
  float r[MAX_EMITTERS];
  float g[MAX_EMITTERS];
  float b[MAX_EMITTERS];
};

__device__ __forceinline__ int Clamp(int v, int n) {
  return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

__device__ __forceinline__ float At(const float* f, int x, int y, int n) {
  return f[Clamp(y, n) * n + Clamp(x, n)];
}

// Bilinear sample with the position kept inside the grid.
__device__ float Sample(const float* f, float x, float y, int n) {
  x = fminf(fmaxf(x, 0.0f), n - 1.001f);
  y = fminf(fmaxf(y, 0.0f), n - 1.001f);
  const int x0 = (int)x;
  const int y0 = (int)y;
  const float tx = x - x0;
  const float ty = y - y0;
  const float* row0 = f + y0 * n;
  const float* row1 = row0 + n;
  const float top = row0[x0] * (1.0f - tx) + row0[x0 + 1] * tx;
  const float bottom = row1[x0] * (1.0f - tx) + row1[x0 + 1] * tx;
  return top * (1.0f - ty) + bottom * ty;
}

// Adds force and dye from every emitter as a gaussian blob.
extern "C" __global__ void ApplyEmitters(float* vx, float* vy, float* red,
                                         float* green, float* blue, int n,
                                         EmitterSet set) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  const int i = y * n + x;
  const float inv_r2 = 1.0f / (set.radius * set.radius);
  for (int e = 0; e < set.count; ++e) {
    const float dx = x - set.x[e];
    const float dy = y - set.y[e];
    const float w = expf(-(dx * dx + dy * dy) * inv_r2);
    vx[i] += set.fx[e] * w;
    vy[i] += set.fy[e] * w;
    red[i] += set.r[e] * w;
    green[i] += set.g[e] * w;
    blue[i] += set.b[e] * w;
  }
}

// Semi-Lagrangian advection: trace each cell back along the velocity and
// take the value found there.
extern "C" __global__ void Advect(const float* src, float* dst, const float* vx,
                                  const float* vy, int n, float dt,
                                  float dissipation) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  const int i = y * n + x;
  const float px = x - dt * vx[i];
  const float py = y - dt * vy[i];
  dst[i] = dissipation * Sample(src, px, py, n);
}

extern "C" __global__ void Curl(const float* vx, const float* vy, float* curl,
                                int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  curl[y * n + x] = 0.5f * ((At(vy, x + 1, y, n) - At(vy, x - 1, y, n)) -
                            (At(vx, x, y + 1, n) - At(vx, x, y - 1, n)));
}

// Vorticity confinement: pushes small swirls back up so the flow does not
// smear out (numerical diffusion from advection kills them otherwise).
extern "C" __global__ void Vorticity(float* vx, float* vy, const float* curl,
                                     int n, float dt, float strength) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  const float gx =
      0.5f * (fabsf(At(curl, x + 1, y, n)) - fabsf(At(curl, x - 1, y, n)));
  const float gy =
      0.5f * (fabsf(At(curl, x, y + 1, n)) - fabsf(At(curl, x, y - 1, n)));
  const float len = sqrtf(gx * gx + gy * gy) + 1e-5f;
  const float c = curl[y * n + x];
  const int i = y * n + x;
  vx[i] += dt * strength * (gy / len) * c;
  vy[i] -= dt * strength * (gx / len) * c;
}

extern "C" __global__ void Divergence(const float* vx, const float* vy,
                                      float* div, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  div[y * n + x] = 0.5f * ((At(vx, x + 1, y, n) - At(vx, x - 1, y, n)) +
                           (At(vy, x, y + 1, n) - At(vy, x, y - 1, n)));
}

// One Jacobi sweep of the pressure Poisson equation, straight from global
// memory: every cell reads its 4 neighbours, so each value is loaded 5 times.
extern "C" __global__ void JacobiNaive(const float* p, float* p_out,
                                       const float* div, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  p_out[y * n + x] =
      0.25f * (At(p, x - 1, y, n) + At(p, x + 1, y, n) + At(p, x, y - 1, n) +
               At(p, x, y + 1, n) - div[y * n + x]);
}

// Same sweep, but the block first copies its 16x16 tile plus a 1-cell halo
// into shared memory, so every pressure value comes from global memory once.
extern "C" __global__ void JacobiTiled(const float* p, float* p_out,
                                       const float* div, int n) {
  __shared__ float tile[TILE + 2][TILE + 2];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int x0 = blockIdx.x * TILE - 1;
  const int y0 = blockIdx.y * TILE - 1;
  for (int k = ty * TILE + tx; k < (TILE + 2) * (TILE + 2); k += TILE * TILE) {
    const int row = k / (TILE + 2);
    const int col = k % (TILE + 2);
    tile[row][col] = At(p, x0 + col, y0 + row, n);
  }
  __syncthreads();
  // Return only after the barrier so edge blocks still help load the halo.
  const int x = x0 + 1 + tx;
  const int y = y0 + 1 + ty;
  if (x >= n || y >= n) return;
  p_out[y * n + x] =
      0.25f * (tile[ty + 1][tx] + tile[ty + 1][tx + 2] + tile[ty][tx + 1] +
               tile[ty + 2][tx + 1] - div[y * n + x]);
}

// Removes the pressure gradient, which leaves a divergence-free velocity.
extern "C" __global__ void SubtractGradient(float* vx, float* vy,
                                            const float* p, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  const int i = y * n + x;
  vx[i] -= 0.5f * (At(p, x + 1, y, n) - At(p, x - 1, y, n));
  vy[i] -= 0.5f * (At(p, x, y + 1, n) - At(p, x, y - 1, n));
}

// Solid walls: no flow through the outer ring of cells.
extern "C" __global__ void SetBoundary(float* vx, float* vy, int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  if (x == 0 || y == 0 || x == n - 1 || y == n - 1) {
    vx[y * n + x] = 0.0f;
    vy[y * n + x] = 0.0f;
  }
}

// Dye -> 8-bit RGB with a soft exposure curve so dense dye does not clip.
extern "C" __global__ void Render(const float* red, const float* green,
                                  const float* blue, unsigned char* rgb,
                                  int n) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= n || y >= n) return;
  const int i = y * n + x;
  rgb[3 * i + 0] = (unsigned char)(255.0f * (1.0f - expf(-red[i])));
  rgb[3 * i + 1] = (unsigned char)(255.0f * (1.0f - expf(-green[i])));
  rgb[3 * i + 2] = (unsigned char)(255.0f * (1.0f - expf(-blue[i])));
}
