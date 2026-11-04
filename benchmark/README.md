# ForceCouplingMethod.jl benchmarks

The scripts in this folder time the package on your computer, on the suspensions the
paper times (Su & Keaveny 2024, §5.2, Fig. 2 and Table 3). They answer two questions:

- **How fast is it on this computer?** (`run.jl`)
- **Did my change make it faster or slower?** (`compare.jl`)

Every change made to speed the package up is checked with them, before and after.

## Before you start

Once, from the repository root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

The graphs (see below) also need Python 3 with matplotlib and numpy.

## How fast is it on this computer?

From the repository root:

```sh
julia -t auto --project=benchmark benchmark/run.jl [label]
```

For each problem (described at the end of this page), in single and then double
precision, the script:

1. **Sets the problem up.**
2. **Warms it up**: at least 50 calls and at least 10 seconds of calls, untimed. The
   first calls of a session are slow while Julia compiles code, and processors and GPUs
   change their clock speed with their load and temperature. On a GPU, 50 calls of a
   small problem last only a fraction of a second, and the card is still changing speed
   when the timings start; ten seconds of continuous calls bring it to the steady state
   that a long simulation runs in.
3. **Times every measurement with exactly 50 calls**, one call per timing. Just before
   them, the measured call runs untimed for one more second. Switching from one kind of
   call to another changes how hard the processor and GPU work, and they take a moment
   to adjust their speed; on calls lasting a millisecond or two, the adjustment
   otherwise shows in the first timed calls.
4. **Frees the problem, then times setup** with 10 calls. Setup is timed last because
   the largest problems need gigabytes of memory, and timing setup needs a second copy.
   Each setup takes up to a few seconds, and a simulation sets up only once.

Then it prints a report and saves it, with every timing, in `benchmark/results/<label>/`.
The label defaults to the date and time. An existing folder is never overwritten, and the
folder is ignored by git, because timings describe one machine.

| File | Contents |
|---|---|
| `report.md` | the readable report, the same as the one printed |
| `machine.txt` | the computer, the software versions, how busy and hot the machine was |
| `timings.json` | every timing, read by `compare.jl` |
| `timings.csv` | every timing as plain text, read by `plot_timings.py` |

**GPU timings** run only when exactly one GPU is visible. On a machine shared with other
people, choose an idle GPU yourself, for example GPU number 6:

```sh
CUDA_VISIBLE_DEVICES=6 julia -t auto --project=benchmark benchmark/run.jl
```

If a problem does not fit in the GPU's free memory, the run skips it and says so. On an
11 GB card this happens to the densest problem in double precision.

**Choosing the problems.** The problems are named by their volume fraction in per cent:
`0.05`, `0.2`, `0.8`, `3.2` and `12.8`. The GPU runs all five by default. The processor
runs `0.05` and `0.2`; the denser problems have 4 to 256 times as many particles and are
too slow on a processor for 50 timed calls of every measurement. The environment
variables `FFCM_BENCHMARK_CPU_PROBLEMS` and `FFCM_BENCHMARK_GPU_PROBLEMS` take a
comma-separated list instead; an empty value skips that part. For example, the GPU part
alone:

```sh
CUDA_VISIBLE_DEVICES=6 FFCM_BENCHMARK_CPU_PROBLEMS= \
    julia --project=benchmark benchmark/run.jl
```

### Reading the report

For every measurement the report gives two numbers per precision:

- **The median**: the middle value of the timings. An occasional slow call, for example
  when another program wakes up, does not move it.
- **The middle half**: the range that holds the middle half of the timings, from the 25th
  to the 75th percentile. A narrow range means steady timings. A wide range means the
  machine was busy or changing speed, and the median deserves less trust.

It also gives **PTPS**, the paper's measure of speed (§5.2): the number of particles
divided by the *average* time of one call. It is given for the whole call and for steps
4–7 alone, the grid and pair work without building the cell list or copying data. The
report ends with the state of the machine at the start and end of each problem: the
processor's load, and the GPU's temperature and clock speed.

What is timed, and why:

- **The whole call to `mobility!`**: the cost of one application of the operator,
  paid on every iteration of a resistance solver.
- **The seven steps of the method** (paper §4), each on its own, to show where the time
  goes and which step a change affected:
  1. wrapping positions into the periodic box;
  2. finding each particle's cell;
  3. sorting particles by cell;
  4. spreading forces onto the grid;
  5. solving the Stokes equations on the grid with FFTs;
  6. interpolating velocities from the grid;
  7. correcting the velocities of nearby pairs.
- **On the GPU only**: the whole call again with the data already in GPU memory, and the
  copies between computer memory and the GPU alone. `mobility!` copies positions and
  forces to the GPU and velocities back on every call, so that users never handle GPU
  arrays. These two measurements show what that costs.
- **Setup** (`FFCMConfig`).
  - On the processor it is timed twice. At setup FFTW, the FFT library, searches for the
    fastest way to transform the grid and remembers the answer for the rest of the
    session, so a second setup of the same grid skips the search. The first measurement
    makes FFTW forget before every timing; the second lets it remember.
  - On the GPU, setup reserves memory and makes the FFT plans. Before each timing, the
    memory and FFT plans of the previous setup are handed back to the GPU, so that every
    timing does all of this work, as a simulation's first setup does. (CUDA.jl
    otherwise keeps them for reuse, which makes a repeated setup faster.)
- **FFT options on the processor**: step 5 with quick FFT plans
  (`fft_planning = :estimate`, no search) against searched plans (the default). If Julia
  was started with more than one thread, also with the FFTs split over all of them
  (`fft_threads`). These numbers back the advice on the documentation's performance page.

## Did my change make it faster?

1. Before the change: `julia -t auto --project=benchmark benchmark/run.jl before`
2. After the change: `julia -t auto --project=benchmark benchmark/run.jl after`
3. Compare:

   ```sh
   julia --project=benchmark benchmark/compare.jl \
       benchmark/results/before benchmark/results/after
   ```

The comparison lists every measurement found in both runs, with its change, and calls it
**faster** or **slower** only when both of these hold:

- the median changed by more than 5 %;
- the middle halves of the two runs' timings do not overlap.

Anything less is within the scatter between runs and is reported as **no real change**.
If the two runs used different machines or software versions, the differences are listed
first, since they can explain a change.

## Graphs

```sh
python3 benchmark/plot_timings.py benchmark/results/before/timings.csv \
    benchmark/results/after/timings.csv --out benchmark/results/plots
```

This draws one image per measurement. Add `--only mobility`, or any other text, to keep
only the measurements whose name contains that text. Each image has two panels:

- **left**: the histograms of the two runs' timings, drawn over each other. Two runs that
  genuinely differ form separate humps; two that only scatter overlap.
- **right**: every timing in the order the calls were made. Drifts, such as a machine
  heating up, show here; a single summary number hides them.

## The test problems

The problems are the suspensions the paper times in §5.2 (Fig. 2 and Table 3): unit
spheres in a periodic cube of side 400 (a/L = 1/400) at five volume fractions. They use
the settings that §5.2 takes from Tables 1–2 for an error of 10⁻⁴:

- the grid spacing equals the width of the computational Gaussian (Σ/Δx = 1.00);
- the Gaussian is spread over M_G = 10 grid points per direction;
- the pair correction reaches R_c = 5.5Σ (λ_ε = 5.5).

The remaining setting, Σ/σ, is Table 3's fastest value for each volume fraction. With
Σ = Δx it fixes the grid, M = L/Σ. The paper does not print its grids, and Table 3 gives
Σ/σ to two figures. For 0.2 % and 0.8 % the grids below are the only ones that
reproduce both of Table 3's values. For the other three the rounding also allows nearby
grids (for the densest suspension, anything from 489³ to 525³), and the benchmark takes
the round sizes:

| Volume fraction | Particles | Grid | Σ/σ (Table 3) | R_c/σ (Table 3) | Timed by default on |
|---|--:|--:|---|---|---|
| `0.05` % | 7,639 | 120³ | 5.91 (5.9) | 32.5 (32.6) | processor, GPU |
| `0.2` % | 30,557 | 140³ | 5.06 (5.1) | 27.9 (27.9) | processor, GPU |
| `0.8` % | 122,230 | 210³ | 3.38 (3.4) | 18.6 (18.6) | GPU |
| `3.2` % | 488,923 | 400³ | 1.77 (1.8) | 9.7 (9.8) | GPU |
| `12.8` % | 1,955,695 | 500³ | 1.42 (1.4) | 7.8 (8.1) | GPU |

The number of particles is rounded down, as the paper does: its Table 1 suspension, 8 %
at a/L = 1/150, has 64,457 particles. For the densest problem Table 3 prints
R_c/σ = 8.1, a little above λ_ε Σ/σ = 7.8; the benchmark follows λ_ε = 5.5, as §5.2
states.

As in the paper, the particles do not overlap. Each one is placed at a random point and
kept only if no particle placed before it is closer than one diameter. Each particle
carries a random force, with every component between −1 and 1, and no torque: like the
paper's §5, the benchmark times the mobility from forces to velocities. A fixed random
seed means every run, and both precisions, time the same particles.

## Things that can mislead you

- **Heat.** Laptops and some desktops slow down by tens of per cent once they are hot.
  Compare runs taken in a similar state, and check the machine state table at the end of
  each report. The warm-up removes the effect of starting cold, but not the effect of a
  machine that is still heating up.
- **Other programs.** On a shared machine, other people's programs compete for the
  processor, memory and GPU. The report records the processor load; prefer quiet times,
  and run the same comparison twice if a result surprises you.
- **Two processor sockets.** Some servers have two processor chips (sockets), each with
  its own memory; `lscpu` shows how many. The operating system decides where a program
  and its memory live, and can move them, and memory on the other socket is slower to
  reach. On such a machine the same measurement can settle at two speeds about 20 %
  apart, differing from one run to the next or jumping in the middle of a measurement (a
  step in the call-by-call graph). Simulations on that machine see the same, so the
  benchmark does not try to prevent it: keep it in mind when comparing runs, and repeat
  a comparison whose result surprises you.
- **Setup times.** A setup is faster when it can reuse memory freed earlier in the
  session, and whether it can depends on what ran before. On the processor, repeated
  setups of the smallest problem took between 10 and 30 ms in different runs, so treat
  changes in setup times with care.
- **Shared GPUs.** Choose an idle GPU (check with `nvidia-smi`). A GPU sitting next to
  busy ones runs warmer, and so can run more slowly.
- **Threads.** `julia -t auto` only affects the "FFTs split over all Julia threads"
  measurement. Everything else runs on one thread, as `mobility!` does by default.

## The reference implementation

[`cufcm-baseline.md`](cufcm-baseline.md) records the speed of cuFCM, the paper authors'
reference CUDA implementation, measured on an RTX 2080 Ti, together with how it was
measured and how it compares.
