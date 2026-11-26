"""Turns the CSV logs in output/ into the charts and image sheets used in the
README and the presentation.

usage: python3 scripts/plot_results.py [output_dir]
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
# Fixed categorical order: naive, tiled, cpu.
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def line_with_label(ax, xs, ys, color, marker, label, dy=0):
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
            label=label)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, dy),
                textcoords="offset points", va="center", color=INK, fontsize=9)


def plot_scaling(out, plots):
    rows = read_csv(out / "scaling.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # The two GPU lines nearly meet at the largest grid; nudge the labels
    # apart so they stay readable.
    for kernel, color, marker, dy in (("naive", BLUE, "o", -9),
                                      ("tiled", ORANGE, "s", 9)):
        sel = sorted((int(r["grid"]), float(r["avg_step_ms"]))
                     for r in rows if r["jacobi_kernel"] == kernel)
        line_with_label(ax, [g for g, _ in sel], [m for _, m in sel], color,
                        marker, f"GPU {kernel} Jacobi", dy)
    cpu = sorted((int(r["grid"]), float(r["cpu_step_ms"]))
                 for r in rows if r["jacobi_kernel"] == "tiled")
    line_with_label(ax, [g for g, _ in cpu], [m for _, m in cpu], AQUA, "^",
                    "CPU, 1 core")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    grids = sorted({int(r["grid"]) for r in rows})
    ax.set_xticks(grids, [f"{g}²" for g in grids])
    ax.set_xlim(grids[0] / 1.3, grids[-1] * 2.2)
    style(ax, "Time per simulation step vs grid size", "grid (cells)",
          "ms per step (log)")
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(plots / "scaling.png", dpi=150)
    plt.close(fig)


def plot_speedup(out, plots):
    rows = [r for r in read_csv(out / "scaling.csv")
            if r["jacobi_kernel"] == "tiled"]
    rows.sort(key=lambda r: int(r["grid"]))
    labels = [f"{r['grid']}²" for r in rows]
    values = [float(r["speedup"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(labels, values, color=BLUE, width=0.55)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.0f}x", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    color=INK, fontsize=9)
    style(ax, "GPU speedup over one CPU core (tiled Jacobi)", "grid",
          "speedup (x)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(plots / "speedup.png", dpi=150)
    plt.close(fig)


def plot_jacobi(out, plots):
    rows = sorted(read_csv(out / "jacobi.csv"),
                  key=lambda r: int(r["jacobi_iters"]))
    iters = [int(r["jacobi_iters"]) for r in rows]
    div = [float(r["div_rms"]) for r in rows]
    ms = [float(r["pressure_ms"]) for r in rows]
    # Two panels instead of one chart with two y-axes.
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.0))
    left.plot(iters, div, color=BLUE, linewidth=2, marker="o", markersize=7)
    left.set_xscale("log", base=2)
    left.set_xticks(iters, [str(i) for i in iters])
    left.set_ylim(0, max(div) * 1.15)
    style(left, "Accuracy: leftover divergence", "Jacobi sweeps per step",
          "RMS divergence (lower = better)")
    right.plot(iters, ms, color=BLUE, linewidth=2, marker="o", markersize=7)
    right.set_xscale("log", base=2)
    right.set_xticks(iters, [str(i) for i in iters])
    style(right, "Cost: pressure solve time", "Jacobi sweeps per step",
          "ms per step")
    fig.tight_layout()
    fig.savefig(plots / "jacobi_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_phases(out, plots):
    rows = read_csv(out / "demo" / "steps.csv")[1:]  # skip warm-up step
    names = ["forces", "advect velocity", "pressure solve", "advect dye"]
    keys = ["forces_ms", "advect_ms", "pressure_ms", "dye_ms"]
    means = [sum(float(r[k]) for r in rows) / len(rows) for k in keys]
    total = sum(means)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bars = ax.barh(names[::-1], means[::-1], color=BLUE, height=0.55)
    for bar, v in zip(bars, means[::-1]):
        ax.annotate(f"{v:.2f} ms ({100 * v / total:.0f}%)",
                    (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=9)
    ax.set_xlim(0, max(means) * 1.35)
    style(ax, "Where one 1024² step spends its GPU time", "ms per step", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(plots / "phases.png", dpi=150)
    plt.close(fig)


def sheet(paths, captions, target, tile=360):
    gap = 8
    caption_h = 28
    img = Image.new("RGB", (len(paths) * (tile + gap) - gap, tile + caption_h),
                    "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for i, (path, caption) in enumerate(zip(paths, captions)):
        img.paste(Image.open(path).convert("RGB").resize((tile, tile)),
                  (i * (tile + gap), caption_h))
        draw.text((i * (tile + gap) + 4, 6), caption, fill=INK)
    img.save(target)


def image_sheets(out, plots):
    frames = sorted((out / "demo" / "frames").glob("frame_*.png"))
    if frames:
        picks = [frames[int(i * (len(frames) - 1) / 5)] for i in range(6)]
        steps = [int(p.stem.split("_")[1]) * 2 for p in picks]
        sheet(picks, [f"step {s}" for s in steps], plots / "demo_keyframes.png",
              tile=300)
        keep = out / "demo" / "keyframes"
        keep.mkdir(exist_ok=True)
        for p, s in zip(picks, steps):
            Image.open(p).save(keep / f"step_{s:04d}.png")
    jac = []
    for iters in (5, 40, 320):
        last = sorted((out / "jacobi" / f"iters_{iters}" / "frames")
                      .glob("frame_*.png"))
        if last:
            jac.append((last[-1], f"{iters} Jacobi sweeps, step 300"))
    if jac:
        sheet([p for p, _ in jac], [c for _, c in jac],
              plots / "jacobi_frames.png")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    if (out / "scaling.csv").exists():
        plot_scaling(out, plots)
        plot_speedup(out, plots)
    if (out / "jacobi.csv").exists():
        plot_jacobi(out, plots)
    if (out / "demo" / "steps.csv").exists():
        plot_phases(out, plots)
    image_sheets(out, plots)
    print(f"plots written to {plots}")


if __name__ == "__main__":
    main()
