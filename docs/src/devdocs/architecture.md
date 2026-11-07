# Architecture

This page documents how the implementation is organised: the two-phase API, the buffer
ownership boundaries, the dispatch seams that carry the CPU/GPU backend split, and the
contracts the hot path is held to. The mathematics of each step lives in the
[method pages](../method/overview.md); this page is about the code.

## Two-phase design

The package is a deep module with two entry points:

- **Cold path** — [`FFCMConfig`](@ref) construction: parameter validation, derived
  scalars, and the allocation of every per-call buffer and FFT plan. Allocations and
  ergonomics are fine here; construction may be slow (FFT planning is a deliberate
  example).
- **Hot path** — [`mobility!`](@ref) (and the [`FFCMMobility`](@ref) `mul!` methods
  built on it): the single call a downstream iterative solver makes per iteration. It
  writes only into configuration-owned buffers and the caller's output, allocates
  nothing, and is fully type-stable.

Every documented precondition is enforced at the outer boundary — the constructor
validates parameters, and `mobility!` guards its argument sizes — so the kernels stay
check-free and their `@inbounds` annotations are justified by construction-time
invariants (for example, the stencil-versus-grid bound ``M_G \leq \min(M_x, M_y, M_z)``
is what makes the spread's inner-loop writes non-aliasing).

## Buffer ownership: the five sub-structs

The compiled buffers are grouped by *kind of information* into five sub-structs, while
the derived scalars stay flat on the configuration:

| Sub-struct | Holds | Backend-specific growth |
|---|---|---|
| `cells` | cell-list integer bookkeeping (hash, sort permutation, per-cell ranges, counting-sort workspace) | the neighbour map — `nothing` on the CPU, a device half-shell map on the GPU |
| `particles` | the ``3 \times N`` wrapped/sorted position and force buffers | host↔device staging buffers — `nothing` on the CPU |
| `grid` | the real-space force-density and fluid-velocity fields | — |
| `solver` | the Fourier-space field, the wavevectors, and the FFT plans | cuFFT plans replace FFTW plans |
| `stencil` | per-particle stencil workspace | unused by the GPU kernels (block-local shared memory replaces it) |

Each sub-struct is parameterized by its storage type, so **the backend is a
type-parameter swap**: a configuration holding device arrays selects device kernels
through ordinary dispatch, and no backend flag is stored or tested at run time. A
backend can carry fields the other does not need (the neighbour map, the staging
buffers) as extra type parameters holding `nothing` on the backend that does not use
them.

## The assembly seam

The constructor runs three phases: validate the parameters, derive the
backend-independent scalars and lookup tables, then assemble the backend buffers and
plans. Only the third phase is backend-specific. The CPU assembly lives in `src`; the
GPU assembly is supplied by the package extension through a method on a `src`-declared
stub whose only `src` method throws — see the
[GPU architecture](gpu-architecture.md) page for how that keeps `src` free of CUDA and
the constructor fully inferred on the CPU path.

## Step functions and function-barrier kernels

Each pipeline step is a public-by-convention wrapper (reachable qualified, e.g.
`ForceCouplingMethod.spread_forces!`, but deliberately not exported — exporting the decomposition would
leak it) that reads the configuration and delegates to a *function-barrier kernel*
taking naked buffers and scalars. The barrier gives the compiler a concrete-typed call
to specialise and lets each kernel be exercised in isolation by the tests. The backend
split happens at these kernels: device methods dispatch on the device buffer types, so
the wrappers — and `mobility!` itself — are identical source for both backends.

## Storage choices

- **Grid fields** (`force_density`, `fluid_velocity`, `fluid_hat`) are
  `StructArray`s of `SVector{3}` over three contiguous component arrays: struct-of-arrays
  memory for stride-1 per-component loops, with a per-grid-point `SVector{3, T}` read
  for consumers. The spread kernel destructures the components and writes plain arrays.
- **Particle data** are column-major ``3 \times N`` matrices, so one particle's three
  components are contiguous.
- **Cell indexing** is `Int32` throughout; realistic cell and particle counts fit
  comfortably.
- **The stencil scratch** (per-axis Gaussian weights, squared distances, wrapped
  indices) is shared between the spread and the interpolation through one fill routine —
  the sharing is what makes interpolation the exact discrete adjoint of spreading (see
  [why the uniform h³ weight](@ref "Why the uniform h³ weight")).
  It is overwritten per particle and carries no between-call invariant.
- **The counting sort** is ``\mathcal{O}(N + m_x m_y m_z)`` and produces the per-cell
  ranges as a byproduct of its prefix sum, with no separate boundary scan; empty cells
  get a well-defined empty range.

## Hot-path contracts

Three guarantees are pinned by the test suite (`test/api/` and `test/hygiene/`), and
changes must keep them green:

- **Zero allocations** on `mobility!` and both `mul!` forms
  (`BenchmarkTools.@ballocated == 0`), scoped to `fft_threads = 1` (threaded FFTW plans
  spawn tasks, which allocate).
- **Inference** (`Test.@inferred`) on every public hot-path function, and JET's
  `@test_opt` pinning the hot path free of runtime dispatch.
- **No mutation of caller inputs**: positions are folded into the configuration-owned
  wrap buffer, never in the caller's `Y`; `mobility!` writes `V` in the caller's
  original particle order.

## Calibration sits on top

[`ForceCouplingMethod.calibrate`](@ref) is a layer above the two-phase API. It only builds
[`FFCMConfig`](@ref)s and calls [`mobility!`](@ref), so it knows nothing about cells,
stencils or kernels, and the hot path does not know it exists. One function,
`_mobility_call`, builds every configuration it times or measures. The rules of what the
constructor accepts stay in `src/config.jl` (`_cutoff_fits_box`, `_support_fits_grid`,
`_has_equal_grid_spacing`, `_validate_fft_options`), where `calibrate` asks them, so it can
only propose settings the constructor would take.

On the GPU it needs one thing the public interface does not give: control of memory. Julia
frees a dropped configuration only when its garbage collector runs, and CUDA.jl keeps freed
memory and FFT plans for reuse, so trying many grids in a row would run out of memory.
`_reclaim_gpu_memory` is declared in `src/calibrate.jl` with a fallback that throws, and the
extension supplies its method, as it does for `_assemble_gpu_buffers`: it runs `GC.gc()`,
then `CUDA.reclaim()`, and reports the memory free. `_mobility_call` calls it before
building each GPU configuration.

`src/calibrate.jl` is organised by what each piece knows, not by the stages of the
[procedure](../choosing-parameters.md), because every stage needs the same knowledge:

| Knowledge | Lives in |
|---|---|
| The suspension and its very accurate answer | `CalibrationProblem` |
| Where the very accurate answer fits, and at what width | `_reference_calculations` |
| What a GPU can hold | `_gpu_grid_bytes`, `_reclaim_gpu_memory` |
| The paper's error measure | `_relative_velocity_error` |
| One accuracy trial | `_measured_error` |
| One timed grid | `_measure_grid` |
| Turning accuracy ratios into `FFCMConfig` settings | `_settings_on_grid` |
| How generous "generous" is | `_generous_ratios`, `_generous_support` |
| Where grid work and pair work balance | `_balanced_grid_spacing`, `_measuring_width` |
| Which grids are worth trying | `_fft_friendly_grids`, `_candidate_settings` |
| That the error only falls as a setting grows | `_smallest_passing` |
| That time against grid is one broad, jagged hump | `_fastest_candidate` |
| How to compare close contenders fairly | `_fastest_taking_turns` |
| How to time a call fairly | `_median_call_time` |
| When to tighten and when to give up | `_tighten_until_met` |

The last five know nothing about the method: they take plain functions and lists, which is
what lets the tests run them on made-up curves. Tuning values are keyword defaults explained
in the docstring of the one function that uses them.

## Internal reference

The step wrappers and buffer sub-structs, in pipeline order:

```@docs
ForceCouplingMethod.wrap_positions!
ForceCouplingMethod.assign_cells!
ForceCouplingMethod.sort_particles_by_cell!
ForceCouplingMethod.spread_forces!
ForceCouplingMethod.stokes_solve!
ForceCouplingMethod.interpolate_velocities!
ForceCouplingMethod.correct_velocities!
ForceCouplingMethod.CellBuffers
ForceCouplingMethod.ParticleBuffers
ForceCouplingMethod.GridBuffers
ForceCouplingMethod.SolverState
ForceCouplingMethod.StencilBuffers
```
