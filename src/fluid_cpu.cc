// Copyright 2026 Parth
#include "src/fluid_cpu.h"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "src/config.h"
#include "src/fluid_common.h"

namespace fluid {
namespace {

using Field = std::vector<float>;

int Clamp(int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); }

float At(const Field& f, int x, int y, int n) {
  return f[static_cast<size_t>(Clamp(y, n)) * n + Clamp(x, n)];
}

float Sample(const Field& f, float x, float y, int n) {
  x = std::fmin(std::fmax(x, 0.0f), n - 1.001f);
  y = std::fmin(std::fmax(y, 0.0f), n - 1.001f);
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const float tx = x - x0;
  const float ty = y - y0;
  const float* row0 = f.data() + static_cast<size_t>(y0) * n;
  const float* row1 = row0 + n;
  const float top = row0[x0] * (1.0f - tx) + row0[x0 + 1] * tx;
  const float bottom = row1[x0] * (1.0f - tx) + row1[x0 + 1] * tx;
  return top * (1.0f - ty) + bottom * ty;
}

void Advect(const Field& src, Field* dst, const Field& vx, const Field& vy,
            int n, float dt, float dissipation) {
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const size_t i = static_cast<size_t>(y) * n + x;
      const float px = x - dt * vx[i];
      const float py = y - dt * vy[i];
      (*dst)[i] = dissipation * Sample(src, px, py, n);
    }
  }
}

void ComputeDivergence(const Field& vx, const Field& vy, Field* div, int n) {
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      (*div)[static_cast<size_t>(y) * n + x] =
          0.5f * ((At(vx, x + 1, y, n) - At(vx, x - 1, y, n)) +
                  (At(vy, x, y + 1, n) - At(vy, x, y - 1, n)));
    }
  }
}

}  // namespace

CpuFluid::CpuFluid(const Config& cfg) : cfg_(cfg), n_(cfg.grid) {
  const size_t cells = static_cast<size_t>(n_) * n_;
  for (Field* f : {&fields_.vx, &fields_.vy, &fields_.red, &fields_.green,
                   &fields_.blue, &vx_next_, &vy_next_, &pressure_,
                   &pressure_next_, &divergence_, &curl_, &dye_next_}) {
    f->assign(cells, 0.0f);
  }
}

void CpuFluid::Step(int step) {
  const int n = n_;
  const float dt = cfg_.dt;
  Field& vx = fields_.vx;
  Field& vy = fields_.vy;

  // 1. Emitters.
  const EmitterSet set = MakeEmitters(step, n, cfg_.force);
  const float inv_r2 = 1.0f / (set.radius * set.radius);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const size_t i = static_cast<size_t>(y) * n + x;
      for (int e = 0; e < set.count; ++e) {
        const float dx = x - set.x[e];
        const float dy = y - set.y[e];
        const float w = std::exp(-(dx * dx + dy * dy) * inv_r2);
        vx[i] += set.fx[e] * w;
        vy[i] += set.fy[e] * w;
        fields_.red[i] += set.r[e] * w;
        fields_.green[i] += set.g[e] * w;
        fields_.blue[i] += set.b[e] * w;
      }
    }
  }

  // Vorticity confinement.
  if (cfg_.vorticity > 0.0f) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        curl_[static_cast<size_t>(y) * n + x] =
            0.5f * ((At(vy, x + 1, y, n) - At(vy, x - 1, y, n)) -
                    (At(vx, x, y + 1, n) - At(vx, x, y - 1, n)));
      }
    }
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        const float gx = 0.5f * (std::fabs(At(curl_, x + 1, y, n)) -
                                 std::fabs(At(curl_, x - 1, y, n)));
        const float gy = 0.5f * (std::fabs(At(curl_, x, y + 1, n)) -
                                 std::fabs(At(curl_, x, y - 1, n)));
        const float len = std::sqrt(gx * gx + gy * gy) + 1e-5f;
        const size_t i = static_cast<size_t>(y) * n + x;
        const float c = curl_[i];
        vx[i] += dt * cfg_.vorticity * (gy / len) * c;
        vy[i] -= dt * cfg_.vorticity * (gx / len) * c;
      }
    }
  }

  // 2. Velocity self-advection.
  Advect(vx, &vx_next_, vx, vy, n, dt, cfg_.velocity_dissipation);
  Advect(vy, &vy_next_, vx, vy, n, dt, cfg_.velocity_dissipation);
  vx.swap(vx_next_);
  vy.swap(vy_next_);

  // 3. Projection.
  ComputeDivergence(vx, vy, &divergence_, n);
  for (int iter = 0; iter < cfg_.jacobi_iters; ++iter) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        pressure_next_[static_cast<size_t>(y) * n + x] =
            0.25f * (At(pressure_, x - 1, y, n) + At(pressure_, x + 1, y, n) +
                     At(pressure_, x, y - 1, n) + At(pressure_, x, y + 1, n) -
                     divergence_[static_cast<size_t>(y) * n + x]);
      }
    }
    pressure_.swap(pressure_next_);
  }
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const size_t i = static_cast<size_t>(y) * n + x;
      vx[i] -= 0.5f * (At(pressure_, x + 1, y, n) - At(pressure_, x - 1, y, n));
      vy[i] -= 0.5f * (At(pressure_, x, y + 1, n) - At(pressure_, x, y - 1, n));
      if (x == 0 || y == 0 || x == n - 1 || y == n - 1) {
        vx[i] = 0.0f;
        vy[i] = 0.0f;
      }
    }
  }

  // 4. Dye advection.
  for (Field* dye : {&fields_.red, &fields_.green, &fields_.blue}) {
    Advect(*dye, &dye_next_, vx, vy, n, dt, cfg_.dye_dissipation);
    dye->swap(dye_next_);
  }
}

}  // namespace fluid
