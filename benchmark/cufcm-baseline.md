# cuFCM reference measurements

cuFCM (<https://github.com/racksa/cuFCM>) is the CUDA implementation of the fast FCM by
the paper's authors (Su & Keaveny 2024, *J. Comput. Phys.* 510, 113060).
ForceCouplingMethod.jl's GPU version was checked against it, for agreement and for speed.
This file records how the comparison was set up and what it found, so that the results
can be read without a copy of cuFCM. The comparison script is a development tool and is
not part of the repository.

## November 2026: the paper's five speed problems

### What was compared

- **Card:** one GeForce RTX 2080 Ti (the model the paper used), on host `nvidia4`.
- **Problems:** the five suspensions of `benchmark/problems.jl`, from the paper's §5.2
  (Fig. 2 and Table 3). Box 400, particle radius 1, volume fractions 0.05 to 12.8 %
  (7,639 to 1,955,695 particles), Σ/Δx = 1, M_G = 10, R_c = 5.5 Σ, grids 120³ to 500³.
  Forces only, no torques.
- **cuFCM:** built with nvcc 12.9 and `-O3 -arch=sm_75`. The upstream makefile compiles
  with `-O0` by default, which would make cuFCM unfairly slow. Linked to cuFFT 11.4.1.4.
  cuFCM takes its settings as three numbers: `alpha` = Σ/Δx = 1, `beta` = M_G/`alpha` = 10
  and `eta` = R_c/Σ = 5.5. Its source was changed in one way only: a timer around each
  whole repetition, with a wait for the GPU at both ends, written to a file for every
  repetition.
- **ForceCouplingMethod.jl 0.1.0:** Julia 1.12.6, CUDA.jl 5.8.5 (CUDA runtime 13.0,
  cuFFT 12.0). Each step of `mobility!` was timed on its own, on data already on the GPU.
  As in cuFCM's timers, the run waits for the GPU after every step.
- **How the runs were arranged:** each code first warmed up for at least 10 s. Then came
  10 alternating blocks, cuFCM first and ForceCouplingMethod.jl second. Each block set its
  code up from scratch, ran it untimed for about 1 s, then timed 50 repetitions. That gives
  500 timed repetitions per code. Alternating means both codes see the same GPU
  temperature and clock. Every time below is a median.
- **Agreement:** the settings cuFCM prints match. Averaged over particles, the two codes'
  velocities differ by 2 to 7 × 10⁻⁷ in single precision and by 2 to 3 × 10⁻¹⁵ in double
  precision (the mean of |ΔV|/|V|, the paper's measure).

### Whole repetition

This is the same job in both codes: positions and forces on the GPU in, velocities in the
original particle order out. It includes the work cuFCM does outside its own step timers.

| Volume fraction | Precision | cuFCM | ForceCouplingMethod.jl | ForceCouplingMethod.jl ÷ cuFCM |
|---|---|--:|--:|--:|
| 0.05 % | single | 1.66 ms | 1.39 ms | 0.84 |
| 0.2 % | single | 3.81 ms | 3.15 ms | 0.83 |
| 0.8 % | single | 14.8 ms | 11.0 ms | 0.74 |
| 3.2 % | single | 71.6 ms | 62.5 ms | 0.87 |
| 12.8 % | single | 262 ms | 188 ms | 0.72 |
| 0.05 % | double | 6.03 ms | 6.16 ms | 1.02 |
| 0.2 % | double | 17.7 ms | 15.5 ms | 0.88 |
| 0.8 % | double | 68.6 ms | 55.9 ms | 0.81 |
| 3.2 % | double | 333 ms | 306 ms | 0.92 |

### Each step

ForceCouplingMethod.jl's time divided by cuFCM's; below 1, ForceCouplingMethod.jl is
faster. The spreading column leaves out ForceCouplingMethod.jl's clearing of the grid,
because cuFCM clears its grid outside its step timers.

| Volume fraction | Precision | Cell list | Spreading | FFT and flow | Interpolation | Correction |
|---|---|--:|--:|--:|--:|--:|
| 0.05 % | single | 0.68 | 1.18 | 1.13 | 1.28 | 0.24 |
| 0.2 % | single | 1.77 | 1.06 | 1.22 | 1.47 | 0.11 |
| 0.8 % | single | 0.96 | 1.06 | 1.18 | 1.42 | 0.08 |
| 3.2 % | single | 0.52 | 1.04 | 1.13 | 1.24 | 0.07 |
| 12.8 % | single | 0.77 | 0.96 | 1.08 | 1.30 | 0.07 |
| 0.05 % | double | 2.49 | 0.58 | 1.06 | 0.68 | 1.40 |
| 0.2 % | double | 2.29 | 0.63 | 1.03 | 0.64 | 0.90 |
| 0.8 % | double | 0.85 | 0.62 | 1.02 | 0.60 | 0.81 |
| 3.2 % | double | 0.66 | 0.98 | 1.01 | 0.60 | 0.82 |

### Why the steps differ

Read from the two codes' source; these are certain:

- **Correction, single precision.** cuFCM's helpers `Q_I` and `Q_xx` divide by
  `Real(4.0)*PI*pow(r, 3)`, where `PI` is a double-precision constant. So every pair does
  part of its arithmetic in double precision, which this card runs at 1/32 of the
  single-precision speed.
- **Clearing the grid.** ForceCouplingMethod.jl clears the grid inside its spreading step.
  cuFCM clears it outside its step timers.
- **Sorting.** cuFCM sorts five lists per particle with a general-purpose GPU sort, two of
  them unused here. Outside its timers, it then sorts all five back into the original
  order. ForceCouplingMethod.jl counts particles per cell, reorders two lists, and writes
  each velocity straight to its original position during interpolation.

Likely causes, not measured on their own:

- **FFT and flow.** Before every complex-to-real transform, CUDA.jl copies the transform's
  input, because cuFFT overwrites it. Estimated from the card's copy speed, the three
  copies per call account for half to all of the single-precision gap. The two codes also
  use different cuFFT versions.
- **Interpolation, single precision.** At every grid point, ForceCouplingMethod.jl
  computes a 64-bit offset into each of the three velocity arrays. cuFCM computes one
  32-bit offset.
- **Cell list at 0.05–0.2 %.** The prefix sum over the cells (CUDA.jl's general-purpose
  scan) costs a fixed 0.1–0.4 ms per call, whatever the number of particles.
- **Correction, double precision, at 0.05 %.** ForceCouplingMethod.jl launches blocks of
  256 threads. With 7,639 particles, that leaves most of the card's 68 multiprocessors
  idle; cuFCM's 32-thread blocks use all of them.
- **Spreading and interpolation, double precision.** cuFCM sets up each particle's stencil
  with double-precision rounding, remainders and divisions, four times as often as
  needed. Its double-precision build also uses 64-bit grid indices.

### Particles per second (PTPS)

The paper's §5.2 defines PTPS as the number of particles divided by the average time to
apply the mobility matrix. cuFCM prints the number of particles divided by the sum of its
spreading, FFT, gathering and correction timers. That leaves out its hashing and sorting
step and the work outside its timers (grid reset, wrapping, sorting back), 0.2–8.4 ms per
repetition here. So its printed PTPS is 2–29 % higher than the whole-repetition figure.

The printed figure on this card, 5.9 to 8.9 million, is close to the paper's Fig. 2
(peaks of about 6.6 to 8.5 million). Measured over the whole repetition in single
precision, cuFCM reaches 4.6 to 8.3 million and ForceCouplingMethod.jl 5.4 to 11.1
million.

### Speed-up

- **As measured:** in single precision ForceCouplingMethod.jl is 1.15–1.39 times as fast
  as cuFCM. In double precision it is 1.09–1.23 times as fast on the three larger
  problems, and 2 % slower on the smallest.
- **Like for like (an estimate):** most of the single-precision lead comes from cuFCM's
  double-precision constant. Suppose cuFCM's correction ran in single precision. Its time
  is estimated from cuFCM's double-precision correction, scaled by ForceCouplingMethod.jl's
  own ratio between the two precisions. Then the two codes would be level at 0.05 %, and
  ForceCouplingMethod.jl would take 6–17 % longer than cuFCM at 0.2–12.8 %, mostly in the
  FFT step and interpolation. The double-precision figures stand as measured.
- **Copies between computer memory and the GPU** are left out above. cuFCM's benchmark
  copies its data once, before it starts repeating. As users call it, with the data in
  computer memory, ForceCouplingMethod.jl's `mobility!` takes 0.2 ms (0.05 %) to 16 ms
  (12.8 %) longer per call in single precision.

### Caveats

- **0.05 %, single precision:** the GPU clock moved between 1350 and 1920 MHz during this
  case. The per-block warm-up was too short, because its length was estimated from a
  warm-up that included Julia's compilation. The ratio ranged from 0.72 to 0.92 between
  blocks; the table gives the median, 0.84. In every other case, the ratio was the same
  in all 10 blocks to within half a per cent.
- **12.8 %, double precision:** skipped. ForceCouplingMethod.jl does not fit in the
  card's 11 GB at this size.
- Measured on one card.

## July 2026 (superseded)

The first comparison used one problem: box 250, a 256³ grid, 186,510 particles, Σ/σ = 2,
M_G = 10 and R_c = 10 σ. It used cuFCM's own timers only, with 50 single-precision and 5
double-precision repetitions, and no warm-up or alternation. It reported
ForceCouplingMethod.jl at "111.7 % of cuFCM's throughput". That figure compared
ForceCouplingMethod.jl's fastest call (20.2 ms, including copies to and from the GPU) with
cuFCM's average hashing-plus-compute time (22.6 ms). Average against average, it was
about 109 % in single precision, and about 108 % in double precision (86.0 against
92.6 ms).

A follow-up ran the cuFCM binary with the newer cuFFT that CUDA.jl shipped at the time.
cuFCM's single-precision FFT time changed by 1.5 %, so the cuFFT version did not explain
the gap. The November measurements replace both results.
