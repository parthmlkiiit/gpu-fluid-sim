# Talk script (about 8–9 minutes)

Read alongside presentation/gpu_fluid_capstone.pptx. The same text is in the speaker notes. Play output/demo/fluid.mp4 on slide 5.

## 1. 

Hi, I'm Parth. This is my capstone for the GPU specialization: a 2-D fluid simulation where every step of the physics runs in CUDA kernels I wrote myself. What you see on the right is one frame of the 1024 by 1024 run. Four coloured jets push dye into still fluid, and the solver turns that into swirls. In the next few minutes I'll cover why I picked this, how one step works, how I compiled the kernels without nvcc, the results, and what surprised me in the profiling.

## 2. 

Three reasons. First, in the previous course project I used NPP, so NVIDIA wrote every kernel. This time I wanted to write all of them myself. Second, I wanted a problem that really needs a GPU. In a fluid solver every cell does the same work every step, and the pressure solve alone sweeps the whole grid forty times per step. Third, I didn't want to just report one speedup number. I wanted to measure where the time goes and what limits it at each size.

## 3. 

This is Stam's Stable Fluids. One step has four phases. First, forces: four gaussian jets add velocity and dye, and vorticity confinement puts back small swirls that the numerical diffusion would otherwise smear out. Second, the velocity advects itself: each cell traces backwards along the flow and takes a bilinear sample. Third, the projection, which makes the fluid incompressible. I compute the divergence, run forty Jacobi sweeps to solve a Poisson equation for pressure, and subtract the pressure gradient. Fourth, the dye is carried by the corrected velocity. Every kernel uses one thread per cell on 16 by 16 blocks. In total that's 51 kernel launches per step, and that number matters later.

## 4. 

A practical problem: my laptop has an RTX 4050 but no Visual Studio, and nvcc needs it on Windows. So instead of compiling the kernels ahead of time, the program reads fluid.cu at startup and compiles it with NVRTC, NVIDIA's runtime compiler, straight to a cubin for this exact GPU. That takes about sixty milliseconds. The host code is plain C++ against the driver API, so it builds with g++ on Linux and on Windows. On the right is the tiled Jacobi kernel. Each block copies its 16 by 16 tile plus a one-cell halo into shared memory, synchronizes, and then does the five-point stencil. One detail: threads outside the grid still help load the halo, and only return after the barrier.

## 5. 

[Play output/demo/fluid.mp4 here, about 15 seconds.] Here's the demo: a 1024 by 1024 grid for 900 steps, with a frame saved every second step. Note how the jets curl into each other and break into smaller vortices. The whole run takes 3.4 milliseconds of GPU time per step, which is 263 times faster than the same solver on one CPU core. Including writing 450 PNG frames, the run takes under 19 seconds. It used to take 161, and I'll explain that in the lessons.

## 6. 

Before trusting any speed numbers, I checked correctness. There's a single-threaded CPU version of the whole solver that mirrors every kernel. Both get exactly the same emitter input, and FMA contraction is off on both sides so the float operations match. After three steps at 1024 squared the largest velocity difference is 8e-5, where velocities go up to about 6. The naive and tiled Jacobi kernels produce byte-identical frames, and I tested that on a 1000 by 1000 grid so the partial edge tiles are covered too. The difference does grow on bigger grids, because expf and sqrtf differ by one ulp and a turbulent flow amplifies tiny differences.

## 7. 

These are the scaling runs: 300 steps per grid size, no frames. On the left, time per step on a log scale. The CPU line grows steadily with the number of cells. The GPU lines are flat at about 0.7 milliseconds up to 512, rise a little at 1024, then jump at 2048. On the right is the speedup: 14x on the smallest grid, 471x at 1024, and down to 207x at 2048. Neither the flat part nor the drop is what I expected, so the next two slides explain them.

## 8. 

Lesson one: small grids are launch-bound. From 128 to 512 squared every step takes about 0.7 milliseconds no matter how many cells there are. That's 51 launches at roughly 13 microseconds each. Making the kernels faster wouldn't change anything; launching fewer would, for example with CUDA Graphs. Lesson two is a cache cliff. Going from 1024 to 2048 is four times the cells but ten times the time. Each Jacobi sweep touches three fields. At 1024 that's 12 megabytes, which fits in the 24-megabyte L2 cache on this GPU, so all forty sweeps run from cache. At 2048 it's 48 megabytes, and every sweep goes to DRAM.

## 9. 

Two things I got wrong at first. Number three: I expected the shared-memory Jacobi to be faster. It wasn't. It was the same or slower, 1.16 versus 0.85 milliseconds at 1024. On this GPU the L1 and L2 caches already give a five-point stencil its neighbour reuse, and the tiled version pays for a halo load and a barrier on every sweep. Tiling pays off when a block reuses its data many times, like matrix multiply. Number four: my first demo run took 161 seconds and reported 13.5 milliseconds per step, instead of 1.9. The cause was encoding PNGs on the main thread. The GPU sat idle for about 180 milliseconds per frame and the laptop dropped its clocks. Moving encoding to a pool of background threads brought the run down to under 19 seconds. The same effect had also made my CPU comparison understate the speedup.

## 10. 

The pressure solve is about two thirds of every step, so I swept the number of Jacobi iterations from 5 to 320. With 5 sweeps the projection is too weak. The fluid stops being incompressible and the dye swells into blobs, like an expanding gas. With 40 you get proper vortices. 320 looks almost the same as 40 but costs seven times as much. The divergence stops improving at about 0.034. That isn't solver error: on this grid layout the five-point Laplacian doesn't exactly match the divergence and gradient operators, so even a perfect solve leaves something behind. A staggered grid would fix that.

## 11. 

Where I'd take this next. First, CUDA Graphs: capture the 51 launches of a step once and replay them, which should remove the launch floor on small grids. Second, a multigrid pressure solver, which converges in a few V-cycles instead of dozens of Jacobi sweeps. Third, a staggered grid to remove the divergence floor. Fourth, OpenGL interop, so the dye is drawn straight from GPU memory in real time instead of going through PNG files. And finally 3-D smoke, where the memory and cache effects would hit much sooner.

## 12. 

To sum up: ten hand-written CUDA kernels, compiled at runtime with NVRTC, running a stable-fluids solver up to 471 times faster than one CPU core. Along the way I learned that launch overhead, cache size and even laptop power management can matter more than the kernel code itself. Everything is in the repository: the code, run scripts, logs, CSVs, plots and the video. Thanks for watching.
