# Benchmarks

This page explains how the package's timings are taken and records measured results, so
that every speed figure in these docs can be traced and reproduced. The scripts live in
the repository's `benchmark/` folder, whose README explains how to run them and how to
compare two runs.

## The problems

The benchmark times the suspensions that §5.2 of the paper times for its Fig. 2 and
Table 3: unit spheres in a periodic cube of side 400 (``a/L = 1/400``) at five volume
fractions. They use the settings that §5.2 takes from Tables 1–2 for an error of
``\varepsilon = 10^{-4}``: ``\Sigma/\Delta x = 1.00``, ``M_G = 10`` and
``\lambda_\varepsilon = R_c/\Sigma = 5.5``. The remaining setting, ``\Sigma/\sigma``, is
Table 3's fastest value for each volume fraction. With ``\Sigma = \Delta x`` it fixes the
grid, ``M = L/\Sigma``. The paper does not print its grids, and Table 3 gives
``\Sigma/\sigma`` to two figures. For 0.2 % and 0.8 % the grids below are the only ones
that reproduce both of Table 3's values. For the other three the rounding also allows
nearby grids (for the densest suspension, anything from 489³ to 525³), and the benchmark
takes the round sizes:

| Volume fraction | Particles | Grid | ``\Sigma/\sigma`` (Table 3) | ``R_c/\sigma`` (Table 3) |
|---|--:|--:|---|---|
| 0.05 % | 7,639 | 120³ | 5.91 (5.9) | 32.5 (32.6) |
| 0.2 % | 30,557 | 140³ | 5.06 (5.1) | 27.9 (27.9) |
| 0.8 % | 122,230 | 210³ | 3.38 (3.4) | 18.6 (18.6) |
| 3.2 % | 488,923 | 400³ | 1.77 (1.8) | 9.7 (9.8) |
| 12.8 % | 1,955,695 | 500³ | 1.42 (1.4) | 7.8 (8.1) |

The number of particles is rounded down, as the paper does (its Table 1 suspension, 8 %
at ``a/L = 1/150``, has 64,457 particles). For the densest suspension Table 3 prints
``R_c/\sigma = 8.1``, a little above ``\lambda_\varepsilon \Sigma/\sigma = 7.8``; the
benchmark follows ``\lambda_\varepsilon = 5.5`` as §5.2 states.

As in the paper, no two particles overlap: each is placed at a random point and kept only
if no particle placed before it is closer than one diameter. Each particle carries a
random force with every component between −1 and 1, and no torque: like the paper's §5,
the benchmark times the mobility from forces to velocities. A fixed random seed means
every run, and both precisions, time the same particles.

## How the timings are taken

- **One problem at a time.** Each problem is set up, warmed up, timed and freed before
  the next, because the largest problems need gigabytes of memory.
- **Warm-up first.** Before a problem is timed it runs, untimed, for at least 50 calls and
  at least 10 seconds. The first calls of a session are slow while Julia compiles code,
  and processors and GPUs change their clock speed with their load and temperature. On a
  GPU, 50 calls of a small problem last only a fraction of a second, and timings taken
  right after them started up to a third slower than they settled at; ten seconds of
  continuous calls bring the machine to the steady state that a long simulation runs in.
- **And again before each measurement.** Just before its timed calls, each measurement
  runs its own call, untimed, for one more second. Switching from one kind of call to
  another changes how hard the processor and GPU work, and they take a moment to adjust:
  without this second, the first 15 or so timings of a 1.5 ms GPU call came out about
  15 % faster than the rest.
- **Exactly 50 timed calls per measurement**, one call per timing. Setup is the exception,
  with 10 timed calls: each takes up to a few seconds, and a simulation sets up once.
- **Median and middle half.** Each measurement is summarised by its median (the middle
  value, which an occasional slow call does not move) and by its middle half. The middle
  half is the range holding the middle half of the timings, from the 25th to the 75th
  percentile. A narrow range means steady timings.
- **PTPS**, the paper's speed measure (§5.2), is the number of particles divided by the
  *average* time of one call.
- **The machine is recorded with the timings**: the processor and GPU, the software
  versions, how busy the processor was, and the GPU's temperature and clock speed at the
  start and end of each problem.

## What is timed

- **The whole call to [`mobility!`](@ref)**: the cost of one application of the operator,
  paid on every iteration of a resistance solver.
- **The seven steps of the method** (paper §4), each on its own, to show where the time
  goes: wrapping positions into the box, finding each particle's cell, sorting particles
  by cell, spreading forces onto the grid, the FFT-based Stokes solve, interpolating
  velocities from the grid, and the correction between nearby pairs.
- **On the GPU**: the whole call again with the data already in GPU memory, and the copies
  between computer memory and the GPU on their own. `mobility!` copies the positions and
  forces to the GPU, and the velocities back, on every call so that users never handle
  GPU arrays; these measurements show what that convenience costs.
- **Setup** ([`FFCMConfig`](@ref)). On the processor it is timed twice, because of
  FFTW's saved plans. At setup FFTW searches for the fastest way to transform the grid
  and remembers the answer for the rest of the session, so a repeated setup skips the
  search. The first measurement makes FFTW forget before every timing; the second lets it
  remember. On the GPU, the memory and FFT plans of the previous setup are handed back to
  the GPU before each timing, so every timing reserves its memory and makes its FFT plans
  from scratch, as a simulation's first setup does.
- **FFT options on the processor**: the Stokes solve with quick FFT plans
  (`fft_planning = :estimate`) against searched plans (`:measure`, the default), and with
  the FFTs split over all Julia threads (`fft_threads`). These back the advice on the
  [Performance tips](../performance.md) page.

## Results

### Processor

Measured on 2026-11-04 on a server with two Intel Xeon Gold 6242 processors (16 cores
each, 64 hardware threads in all) under Linux, with Julia 1.12.6, FFTW 1.10.0 and
ForceCouplingMethod 0.1.0, started with `julia -t auto`: one Julia thread per call except
where the FFTs were split over all 64 threads. The two most dilute problems were timed;
the denser ones have 4 to 256 times as many particles (on 210³ to 500³ grids) and are too
slow on a processor for 50 timed calls of every measurement. The machine was shared with
other users' programs (1–10 cores busy on average, this run included). For almost every
measurement the middle half of the timings lay within 2 % of the median; the FFTs split
over all threads, which compete with other programs for every core, scattered more.

| Volume fraction | Float32 whole call | Float64 whole call | Float32 PTPS | Float64 PTPS |
|---|--:|--:|--:|--:|
| 0.05 % (120³ grid, 7,639 particles) | 108 ms | 153 ms | 70,500 | 49,900 |
| 0.2 % (140³ grid, 30,557 particles) | 325 ms | 420 ms | 94,000 | 72,700 |

Where the time goes (medians). At the most dilute suspension the Stokes solve is the
largest cost. At 0.2 %, with four times as many particles on a grid only 1.6 times
larger, spreading and interpolation cost as much as the Stokes solve. Building the cell
list (steps 1–3) takes well under 1 % of the call. Table 3's values of ``\Sigma/\sigma``
were tuned for the paper's GPU; on a processor the fastest values can differ.

| Step | 0.05 %, Float32 | 0.05 %, Float64 | 0.2 %, Float32 | 0.2 %, Float64 |
|---|--:|--:|--:|--:|
| 1–3. Build the cell list | 0.21 ms | 0.23 ms | 0.85 ms | 1.07 ms |
| 4. Spread forces onto the grid | 25.0 ms | 37.2 ms | 94.2 ms | 109 ms |
| 5. Stokes solve on the grid (FFTs) | 46.4 ms | 75.8 ms | 88.4 ms | 160 ms |
| 6. Interpolate velocities from the grid | 21.1 ms | 28.5 ms | 82.9 ms | 99.7 ms |
| 7. Correct nearby pairs | 7.10 ms | 7.15 ms | 52.6 ms | 50.2 ms |

Setup and FFT options. A first setup includes FFTW's search for fast FFT plans; a
repeated setup of the same grid in the same session reuses the saved result. The Stokes
solve is timed with quick plans (`fft_planning = :estimate`), with searched plans (the
default), and with the FFTs split over all 64 threads (`fft_threads = 64`).

| Volume fraction | Precision | Setup, first | Setup, repeated | Stokes solve, quick plans | searched plans | 64 threads |
|---|---|--:|--:|--:|--:|--:|
| 0.05 % | Float32 | 659 ms | 12.9 ms | 71.7 ms | 46.4 ms | 15.2 ms |
| 0.05 % | Float64 | 856 ms | 24.6 ms | 101 ms | 74.4 ms | 17.0 ms |
| 0.2 % | Float32 | 459 ms | 19.1 ms | 110 ms | 86.0 ms | 25.2 ms |
| 0.2 % | Float64 | 654 ms | 33.6 ms | 189 ms | 195 ms (see below) | 27.7 ms |

On these grids, searching for FFT plans made the Stokes solve 15–35 % faster, for a
one-time search of 0.4–0.8 s, and splitting the FFTs over 64 threads made it 3–6 times
faster. For double precision at 0.2 % these figures use the main measurement of the same
solve with searched plans (160 ms, in the table of steps above).

Two caveats, both from the machine rather than the code:

- **The same work at two speeds.** This server has two processor sockets, each with its
  own memory, and the operating system decides where a program's memory lives. The
  double-precision Stokes solve at 0.2 % ran at about 160 ms in the main measurement
  and about 195 ms in the searched-plans measurement, which does exactly the same work.
  In an earlier run the main measurement jumped from 160 to 196 ms partway through,
  while the searched-plans one stayed at 158 ms. Four separate FFT plan searches all gave
  157–161 ms, so the plan is not the cause; where the memory sits most likely is. The
  benchmark does not try to prevent this, because simulations on such a machine see the
  same.
- **Setup times vary between runs.** A setup is faster when it can reuse memory freed
  earlier in the session, and whether it can depends on what ran before. The repeated
  setup of the smallest problem took 13 ms in this run and between 10 and 30 ms in
  earlier ones. The first setup, dominated by FFTW's search, varied by less than 5 %.

### GPU

Measured on 2026-11-04 on an NVIDIA GeForce RTX 2080 Ti, the GPU model of the paper's
Fig. 2, in the same server, with driver 595.84, CUDA.jl 5.8.5 (CUDA runtime 13.0) and
Julia 1.12.6. All five problems were timed in single precision, and the four that fit in
the card's 11 GB in double precision.

| Volume fraction | Float32 whole call | Float32 PTPS | Float32 PTPS, data on the GPU | Float64 whole call | Float64 PTPS |
|---|--:|--:|--:|--:|--:|
| 0.05 % (120³ grid, 7,639 particles) | 1.69 ms | 4,610,000 | 5,290,000 | 6.37 ms | 1,200,000 |
| 0.2 % (140³ grid, 30,557 particles) | 3.57 ms | 8,550,000 | 9,550,000 | 15.8 ms | 1,930,000 |
| 0.8 % (210³ grid, 122,230 particles) | 12.2 ms | 10,000,000 | 11,400,000 | 57.7 ms | 2,120,000 |
| 3.2 % (400³ grid, 488,923 particles) | 66.4 ms | 7,370,000 | 7,850,000 | 314 ms | 1,560,000 |
| 12.8 % (500³ grid, 1,955,695 particles) | 206 ms | 9,520,000 | 10,300,000 | — | — |

The whole call is timed as users see it, with the positions, forces and velocities in
computer memory. The paper defines PTPS as the number of particles divided by the average
time to apply the mobility matrix (§5.2). Read off its Fig. 2 (the same GPU model, single
precision), the peak values range from about 6.6 million PTPS at 0.05 % to about 8.5
million. With the data already on the GPU, ForceCouplingMethod.jl reaches more at 0.2,
0.8 and 12.8 % (9.6, 11.4 and 10.3 million) and less at 0.05 and 3.2 % (5.3 and 7.9
million); at 0.05 % a call takes under 1.5 ms, and the fixed cost of each call weighs
most. The paper's figures come from its own code and the software of its time, and Table
3's settings were tuned for that code, so this is only a rough comparison. Double
precision is 4 to 5 times slower than single on this card: GeForce cards do
double-precision arithmetic far more slowly than single.

Where the time goes (medians, Float32, data on the GPU). The Stokes solve is the largest
cost at every volume fraction, 38–48 % of the call, followed by spreading; the pair
correction stays below 5 %. With the data in computer memory, `mobility!` also copies the
positions and forces to the GPU and the velocities back on every call, which makes up
6–13 % of the call as users see it; keeping the data on the GPU avoids it, since
`mobility!` also accepts GPU arrays.

| Step | 0.05 % | 0.2 % | 0.8 % | 3.2 % | 12.8 % |
|---|--:|--:|--:|--:|--:|
| 1–3. Build the cell list | 0.09 ms | 0.10 ms | 0.12 ms | 0.50 ms | 4.65 ms |
| 4. Spread forces onto the grid | 0.24 ms | 0.81 ms | 3.27 ms | 18.1 ms | 51.7 ms |
| 5. Stokes solve on the grid (FFTs) | 0.70 ms | 1.20 ms | 4.04 ms | 30.1 ms | 82.3 ms |
| 6. Interpolate velocities from the grid | 0.18 ms | 0.69 ms | 2.92 ms | 12.4 ms | 45.1 ms |
| 7. Correct nearby pairs | 0.06 ms | 0.12 ms | 0.39 ms | 0.99 ms | 5.58 ms |
| Copies between computer memory and the GPU | 0.08 ms | 0.29 ms | 1.02 ms | 3.59 ms | 15.1 ms |

A setup from scratch took from 3.4 ms at the most dilute suspension to 131 ms at the
densest, in single precision. Table 3's values of ``\Sigma/\sigma`` were tuned for the
paper's code; ForceCouplingMethod.jl's fastest values can differ.

One caveat: the smallest problem scatters more. Its calls take under 2 ms, and every 8 or
9 calls one or two run about 0.1 ms faster or slower, so the middle half of its whole call
lies within 5 % of the median. For every other whole call it lies within about 1 %.

### Calibration

The times under [What it costs](@ref "What it costs") were measured on 2026-11-06 on the
same server and software, on one Julia thread, with other programs running on the machine.
Each is the first call of [`ForceCouplingMethod.calibrate`](@ref) in a new Julia session, so
it includes compiling it, and the time of one call comes from the report it logs. To repeat
one, from the repository root with `julia --project=benchmark`:

```julia
using ForceCouplingMethod
include("benchmark/problems.jl")

problem = benchmark_problem("0.05")
Y, F = paper_suspension(Float32, problem)
L = (Float32(BOX), Float32(BOX), Float32(BOX))
seconds = @elapsed settings = ForceCouplingMethod.calibrate(Y; L, F)
```

The [saving from the starting grid](@ref "The starting grid") was measured the way the test
suite checks it, in `test/accuracy/test_calibrate.jl`: each grid is timed once, and both
walks are given the same times, so timing noise can't decide the comparison. Those runs were
in `Float32` too.
