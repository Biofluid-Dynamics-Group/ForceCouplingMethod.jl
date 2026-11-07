# Test suite guide

Tests are grouped into subdirectories, included by `runtests.jl` in pipeline
order (accuracy and API files interleave step by step):

- **`accuracy/`** (`test_<step>.jl`) pins physical and mathematical
  behavior against an oracle that is independent of the implementation.
- **`api/`** (`test_<step>_api.jl`) pins the two machine-level guarantees
  of the hot path: type stability (`Test.@inferred`) and zero heap
  allocations (`BenchmarkTools.@ballocated` with `samples=1 evals=1`; the
  counts are deterministic after warmup, so one sample suffices). They test
  the step *entry points* only — a zero-allocation entry bounds its callees,
  and JET walks inference into them — so internal kernels are deliberately
  not exercised here. `test_calibrate_api.jl` is the exception: calibration is
  cold-path code, so it pins what calibration takes, returns, logs and refuses.
- **`hygiene/`** (`test_aqua.jl`, `test_jet.jl`, `test_exported_surface.jl`,
  `test_ascii_public_surface.jl`) audits the package as a whole.
- **`cuda/`** (`test_gpu_*.jl`) pins the CUDA backend: backend-selection
  errors (no device needed), CPU↔CUDA parity (one device required), and
  calibration on the GPU (`test_gpu_calibrate.jl`). See *Running the GPU tests*
  below for how these are guarded.

Shared fixtures live in `test_utilities.jl`, directly under `test/`
(standard config builder, clustered particle cloud, near-zero tolerance),
included before everything else.

## Tolerance convention

`sqrt(eps(T))` is the **relative** tolerance for comparisons against a
non-zero reference. Quantities that should be `≈ 0` use the **absolute**
floor `_near_zero_atol(T)` (`1e-10` for `Float64`, `1e-6` for `Float32`).
Truncation-dominated tests (grid-resolution limited, not round-off limited)
document their own physical tolerance inline.

## Running the GPU tests

The `cuda/` parity tests run only when **exactly one CUDA device is visible**.
On a single-GPU machine that is automatic — `Pkg.test()` finds the card and runs
them. On a shared, queue-less node with several GPUs, pick a free card and pin it
first, so the run cannot touch a card another user is on:

```bash
gpustat                         # or nvidia-smi — find a free card
export CUDA_VISIBLE_DEVICES=<index>
julia --project=. -e 'using Pkg; Pkg.test()'
```

`CUDA_VISIBLE_DEVICES` is a CUDA driver feature that masks every other GPU, so the
process physically cannot allocate on them and the pinned card appears as device 0
(CUDA.jl honors it natively — no ForceCouplingMethod.jl code selects a device). The suite prints
the selected card (name + free memory) before the GPU tests; if it reports several
GPUs visible and none pinned, it skips the GPU tests and prints this recipe. The
backend-selection test (`test_gpu_backend_selection.jl`) needs no device and always
runs. It also checks, with a stand-in for the CUDA extension that builds on the
processor, that a calibration for the GPU sends every calculation there and frees
memory before each one, so that this is tested on machines without a GPU too.

## What each file pins

| File | What it pins | Oracle / justification |
| --- | --- | --- |
| `accuracy/test_cell_geometry.jl` | Cell counts `max(L_i/R_c, 3)`, sizes, buffer shapes, constructor rejections | Paper §4 cell-list formula, hand-computed |
| `accuracy/test_wrap_positions.jl` | `[0, L)` fold: boundary, idempotence, integer-period invariance | Closed-form `mod` |
| `accuracy/test_assign_cells_kernel.jl` | Hash linearisation `x + (y + z⋅m_y)⋅m_x`, upper-edge clamp, hash bounds | Hand-computed hashes |
| `accuracy/test_assign_cells.jl` | Entry point delegates to the kernel on config fields and returns the config buffer | Kernel called directly |
| `accuracy/test_sort_particles_by_cell.jl` | Counting sort: ordering, cell bracketing, stability, gather permutation | Permutation identities |
| `accuracy/test_fcm_grid.jl` | Constructor invariants: `σ = a/√π`, `Σ = (Σ/σ)⋅σ`, isotropic `h`, `2 ≤ M_G ≤ min(M)`, SoA buffers; inferred-`T` form matches the explicit form and rejects non-float `eltype(L)` | Paper §2/§3/§5 relations |
| `accuracy/test_modified_kernel_coefficients.jl` | Eq. (22) expansion scalars; `Σ = σ` degenerate limit | Independent algebra of the same expansion |
| `accuracy/test_spread_forces.jl` | Spread matches the closed-form modified kernel; anchor convention; force conservation; first moment; translation; linearity; `Σ = σ` collapse | Paper §3 eq. (22) closed form; integral identities |
| `accuracy/test_stokes_solve.jl` | Wavenumber layout; `k = 0` gauge fix; per-mode incompressibility; analytical single mode; linearity; translation/reflection equivariance; `1/μ` scaling | Analytical Fourier solutions and symmetries |
| `accuracy/test_interpolate_velocities.jl` | Adjoint identity `J = h³⋅Sᵀ`; original-order output; closed-form gather; constant-flow limit; linearity; translation; assembled-operator symmetry | Paper §3; closed-form sums |
| `accuracy/test_correct_velocities.jl` | Pair correction `A⋅I + B⋅xxᵀ` and self term; `Σ = σ` zero limit; symmetry; linearity; translation/periodic invariance; accumulation into `V` | Full-tensor difference-of-mobilities (paper §2 eqs. (8)–(10), (16)–(17); §3 eqs. (30)–(31); App. B eq. (B.1)) — an algebraically distinct route. Deliberately tests the internal scalars too: they *are* the algebraic collapse under test |
| `accuracy/test_mobility.jl` | Driver contracts: caller arrays untouched, `DimensionMismatch` guards, linearity, SPD, `mul!` (3- and 5-arg), `M * F`, `issymmetric`/`isposdef` | Operator identities; the `mobility!`/`FFCMMobility` docstring contracts |
| `accuracy/test_mobility_properties.jl` | Linearity, symmetry, positivity across seeded-random domains, grids, widths, and out-of-domain positions | Exact operator identities, reproducible RNG |
| `accuracy/test_single_sphere_mobility.jl` | **Headline**: end-to-end self-mobility against the reciprocal-lattice regularised-Stokeslet sum; Σ-independence of the assembled operator | Paper §3 eqs. (32)–(33) lattice sum (`Float64` only: the bound is grid-truncation, which `Float32` round-off would mask) |
| `accuracy/test_calibrate.jl` | The building blocks (error measure, searches, grids, settings, reference) and the whole calibration: tolerance met without overdoing it, no other grid meaningfully faster, the starting ansatz beats starting from the coarsest grid. Small problems, so that a calibration takes seconds | Hand-worked values and made-up curves for the blocks; standard FCM computed independently (`Σ = σ` on a fine grid, paper §3) for the tolerance |
| `api/test_*_api.jl` (x7) | `@inferred` + zero allocations for each step entry point and the assembled operator | The hot-path contracts stated in each step's docstring |
| `api/test_calibrate_api.jl` | What `calibrate` returns in either precision, its defaults, inputs left untouched, the report, and every refusal | The `calibrate` docstring |
| `hygiene/test_exported_surface.jl` | Exports are exactly `FFCMConfig`, `mobility!`, `FFCMMobility`; `calibrate` is public but not exported | The two-phase public surface (`docs/src/devdocs/architecture.md`; the `FFCMConfig`/`mobility!` docstrings) |
| `hygiene/test_ascii_public_surface.jl` | Exported names and public keyword arguments are ASCII (typeable in any terminal); Unicode stays internal | Method introspection (`Base.kwarg_decl`), with sentinel keywords against vacuous passes |
| `hygiene/test_aqua.jl` | Package hygiene | Aqua.jl (below) |
| `hygiene/test_jet.jl` | Whole-call-graph inference and dispatch health | JET.jl (below) |

## Why JET and Aqua

**JET.jl** statically analyses the *inferred call graph* from an entry point
— it follows every reachable callee, which `@inferred` (return type of one
call) cannot do.

- `@test_call f(args...)` fails on inference **errors** anywhere in the
  graph (method errors, undefined fields, unbound variables). We run it on
  every hot-path entry point *and* the cold-path constructor.
- `@test_opt f(args...)` additionally fails on any **runtime dispatch**
  remaining in optimized code. Runtime dispatch usually allocates, so this
  is the static counterpart of the `@ballocated == 0` guarantee. It runs on
  the hot path only — the cold constructor is allowed to be dynamic — and is
  scoped with `target_modules = (ForceCouplingMethod,)` so FFTW internals do not produce
  foreign noise.

**Aqua.jl** (`Aqua.test_all`) audits package hygiene: stale `[deps]`, method
ambiguities, type piracy, unbound type parameters, undocumented exports, and
`persistent_tasks`.

## Known gaps (deliberate, deferred)

- No anisotropic-`L` accuracy test through the assembled operator (the
  property tests vary the grid but keep `h` isotropic, as the paper assumes).
- No two-sphere pair end-to-end test against a published reference value
  (planned alongside `examples/`).
- The end-to-end Stokeslet-shape test in `test_stokes_solve.jl` is
  qualitative; the quantitative single-sphere case lives in
  `test_single_sphere_mobility.jl`.
