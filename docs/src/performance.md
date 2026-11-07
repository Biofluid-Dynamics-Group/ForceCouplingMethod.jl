# Performance tips

The package follows a two-phase design: construction ([`FFCMConfig`](@ref)) may be slow
and allocate freely; the operator application ([`mobility!`](@ref) and the
[`FFCMMobility`](@ref) `mul!` methods) is allocation-free and meant to run many times —
once per iteration of a downstream iterative solver. The knobs below tune that trade.

## Choose the settings for your problem

The largest lever is the method's own settings: the fastest grid can differ a lot from one
suspension or machine to another. [`ForceCouplingMethod.calibrate`](@ref) finds the fastest
settings that meet an error you choose, on your own particles and machine, as explained on
the [Choosing parameters](choosing-parameters.md) page. Pass it the same `fft_planning`,
`fft_threads` and `gpu_acceleration` you will run with, because they change which grid is
fastest.

## Warm up before timing

The first `mobility!` (or `M * F`) call after loading the package incurs Julia's
just-in-time compilation. For timing or long solves, make one warmup call on the
configuration first; subsequent calls run at full speed.

## FFT planner effort (`fft_planning`)

The `fft_planning` keyword (`:estimate`, `:measure` — the default — or `:patient`)
selects the FFTW planner effort. Planner effort trades construction time for transform
time: `:estimate` plans immediately from heuristics, while `:measure` and `:patient`
time candidate algorithms on the actual grid at construction and can produce faster
plans for the many Stokes solves of a long resistance solve. On the benchmark machine's
120³ and 140³ grids (see [Benchmarks](devdocs/benchmarks.md)), `:measure` plans ran the
Stokes solve 15–35 % faster than `:estimate`, for a one-time planning cost of 0.4–0.8 s.
Choose `:estimate` when construction latency matters more than per-call speed — small
grids, one-shot evaluations. The solution is independent of the planner effort to
round-off.

## FFTW wisdom

FFTW can cache planner results across sessions through its wisdom mechanism, with no
ForceCouplingMethod involvement: call `FFTW.import_wisdom(path)` before constructing the
configuration
and `FFTW.export_wisdom(path)` after. A wisdom-loaded construction gets measured-quality
plans at `:estimate`-like planning cost: on the benchmark machine a setup on a 120³ or
140³ grid took tens of milliseconds with saved plans instead of 0.46–0.86 s without.

## FFT threading (`fft_threads`)

The `fft_threads` keyword (default 1) bakes a thread count into the two FFT plans.
Threaded transforms pay off on large grids: with 64 threads the Stokes solve ran 3–6
times faster on 120³ and 140³ grids (see [Benchmarks](devdocs/benchmarks.md)). On small
grids, coordinating the threads can cost more than the transform itself, so measure
before opting in.
Two caveats: threaded FFTW plans execute as spawned Julia tasks, which allocate on every
call, so the allocation-free hot-path guarantee is scoped to `fft_threads = 1`; and the
thread count is per-plan state, so it leaves no global FFTW state behind. The solution
matches the single-threaded result to round-off.

## The cell list is rebuilt every call

`mobility!` rebuilds the cell list (wrap → hash → sort → gather) on every call, even
when the same positions are reused across a linear solve. This keeps the public contract
a single call and is cheap: the cell-list build is ``\mathcal{O}(N)``, dominated by the
spread and interpolation (``\mathcal{O}(N M_G^3)``) and the FFT
(``\mathcal{O}(M \log M)``) that must run every call because the forces change.

## GPU

For large problems the [CUDA backend](gpu.md) is the main lever: construct with
`gpu_acceleration = true` and use `Float32`. The same warmup advice applies — the first
call additionally JIT-compiles the device kernels.
