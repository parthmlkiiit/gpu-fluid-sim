// Copyright 2026 Parth
#include "src/fluid_common.h"

#include <cmath>

namespace fluid {
namespace {

constexpr float kPi = 3.14159265f;

// Orange, cyan, magenta, lime.
constexpr float kColors[kMaxEmitters][3] = {
    {1.0f, 0.45f, 0.1f},
    {0.1f, 0.7f, 1.0f},
    {0.95f, 0.15f, 0.75f},
    {0.45f, 1.0f, 0.2f},
};

}  // namespace

EmitterSet MakeEmitters(int step, int n, float force) {
  EmitterSet set = {};
  set.count = kMaxEmitters;
  set.radius = 0.018f * n;
  const float center = 0.5f * n;
  const float ring = 0.32f * n;
  const float t = static_cast<float>(step);
  // Forces are given for a 512 grid; scale so the motion looks the same in
  // cells-per-step terms at any resolution.
  const float strength = force * n / 512.0f;

  for (int e = 0; e < kMaxEmitters; ++e) {
    const float angle = e * 0.5f * kPi + 0.004f * t;
    set.x[e] = center + ring * std::cos(angle);
    set.y[e] = center + ring * std::sin(angle);

    // Point at the centre, turned 0.9 rad so the jets curl, plus a wobble.
    const float aim = angle + kPi + 0.9f + 0.5f * std::sin(0.05f * t + e);
    const bool on = std::sin(0.02f * t + 1.3f * e) > -0.3f;
    const float push = on ? strength : 0.0f;
    set.fx[e] = push * std::cos(aim);
    set.fy[e] = push * std::sin(aim);

    const float dye = on ? 0.25f : 0.0f;
    set.r[e] = dye * kColors[e][0];
    set.g[e] = dye * kColors[e][1];
    set.b[e] = dye * kColors[e][2];
  }
  return set;
}

}  // namespace fluid
