# CLAUDE.md — agent operating guide for ForceCouplingMethod.jl

This file tells Claude (and any other coding agent) how to work on this
repository.

## 1. What this package is

ForceCouplingMethod.jl is a Julia implementation of the Fast Force-Coupling Method (FFCM)
for hydrodynamic interactions between rigid particles in a triply-periodic
Stokes flow, following Su & Keaveny (2024), *J. Comput. Phys.* 510, 113060.

FFCM is a
particular approximation to the mobility operator $\mathcal{M}^{\mathcal{V}\mathcal{F}}$
that maps a finite number of forces $\mathcal{F}$ in spherical particles of radius $a$ localised at positions $\mathcal{Y}$ to
their corresponding velocities $\mathcal{V}$ if they interact hydrodynamically via the
Stokes equation. This can be the Stokes solution to a problem of colloidal suspension, or, equivalently, an immersed-boundary method where the distributional terms have been regularised using Gaussian kernels corresponding to the hydrodynamic radius.

The package exposes a plug-and-play mobility operator

```julia
mobility!(V, config, Y, F)   # Y, F: 3xN arrays; particles have unit radius
```

with two backends:

- **CPU**: lives entirely in `src/`.
- **CUDA (for NVIDIA GPUs)**: lives in `ext/ForceCouplingMethodCUDAExt/` and loads as a package
  extension only when the user has `CUDA.jl` in their environment.

The intended downstream use is a direct or iterative resistance solver that applies
the mobility operator (potentially many times per linear solve), so the hot path is
allocation-free and the GPU backend hides host-device traffic from callers.

## 2. Ground rules

1. **The paper is the spec.** Notation, equations, and parameter meanings
   in this codebase follow Su & Keaveny (2024) unless otherwise stated. The mapping between paper
   symbols and code identifiers lives in
   [docs/src/notation.md](docs/src/notation.md).

2. **The port is self-contained.** The original C++/CUDA implementation by
   the paper's authors (racksa/cuFCM) served as the implementation
   reference for the GPU backend; numerical parity was validated
   (2026-07-13) and shipped artefacts no longer cite it beyond the tracked
   baseline record `benchmark/cufcm-baseline.md` and one acknowledgment
   each in `README.md` and the developer docs (see §6).

3. **`docs/` and docstrings are the source of truth for *what the code
   does*.** The paper is the source of truth for *why*. Contracts —
   arguments, shapes, units, preconditions — live in `src/` docstrings;
   method narrative lives in the Documenter pages under `docs/src/`,
   citing the paper by section/equation rather than restating it. No new
   public API or algorithmic component is added without (a) its contract
   written into the docstring (plus a docs-page section where the method
   warrants narrative) and (b) a failing test that drove the
   implementation (see §5).

4. **Hot-path code is allocation-free.** Any function reachable from
   `mobility!` after the `config` is built must not allocate on the heap.
   The per-step `test/test_*_allocations.jl` files enforce this with
   `BenchmarkTools.@ballocated`.

5. **Float32 is the default; Float64 must also work.** Code is
   type-parametric on a `T <: AbstractFloat`. Tolerances in tests scale
   with `T`. `sqrt(eps(T))` is the baseline **relative** tolerance only;
   use it for `rtol` when comparing to a non-zero reference. For quantities
   that should be `≈ 0`, or where a relative tolerance is ill-defined, use
   an **absolute** tolerance: `atol = 1e-10` for `Float64`, `1e-6` for
   `Float32` (≈ `rtol/100`). Combine both (`isapprox(a, b; rtol, atol)`)
   when an array mixes large and near-zero entries. Truncation-dominated
   tests use the documented physical-error tolerance, not the round-off
   one. Relax any tolerance only with a documented numerical reason.

6. **CPU first, CUDA second.** A new feature is implemented and tested as normal CPU code before any CUDA work begins. Once the CPU version passes
   its accuracy tests, the CUDA implementation must pass a CPU↔CUDA parity
   test before it is considered done.

7. **Every documented precondition is enforced.** If a docstring or
   docs-page contract states a precondition (`M_G ≥ 2`,
   `M_G ≤ min(num_grid_points)`, `Y` is `3xN`, `R_c ≤ min(L)/2`), the
   cold-path constructor or the hot-path entry validates it. A
   precondition that the code relies on (especially anything guarded by
   `@inbounds` or asserted by `@simd`) but does not check is a bug, not a
   convenience. Validate at the outer boundary so kernels stay
   check-free.

8. **Cite the published paper, by section/equation.** Shipped artefacts —
   `src/` docstrings and `docs/` pages — reference Su & Keaveny (2024),
   *J. Comput. Phys.* 510, 113060 by section and numbered equation. A
   Method docs page leads with a paper-math summary a paper-literate,
   Julia-naive reader can follow; types and performance notes live in the
   Developer documentation section (`docs/src/devdocs/`) or as code
   comments at the decision site.

9. **Docstrings say what the function does.** A body that states the
   function's job (with the paper citation), then `# Arguments` and `# Returns` sections
   wherever possible — each entry gives at least the type and/or size
   plus any real contract (valid range, units, ordering). Type/size
   information lives in those sections, not in the body. No boilerplate
   guarantees ("allocation-free", "type-stable on `T`") and no
   self-justifying notes; extra notes survive only where they aid the
   reader — e.g. the precondition lists that justify `@inbounds`.

## 3. Design boundaries

The FFCM state object is called `config` (e.g. `FFCMConfig`). It owns
both user-set parameters and compiled state.

**Two-phase public API.**

- **Cold path** — construction and tuning: `FFCMConfig(...)`, parameter
  accessors, and any rebuild helpers triggered by a grid or particle-count
  change. Allocations are fine here; ergonomics matter.
- **Hot path** — operator action: `mobility!(V, config, Y, F)` only.
  This is the single call made per iteration of the downstream iterative
  solver. Do **not** add
  logging-decorated variants, default-config helpers, or any wrapper that
  bypasses `config` on the hot path.

This is the deep-module design: `config` plus `mobility!` together hide
spreading, FFT, Stokes solve, real-space correction, and interpolation
behind one cold constructor and one hot call.

**Export only the public surface.** The module exports `FFCMConfig`,
`mobility!`, and `FFCMMobility` — nothing else. The seven internal step
functions (`spread_forces!`, `stokes_solve!`, …) stay reachable as
`ForceCouplingMethod.spread_forces!` for tests and advanced use, but exporting them
would leak the internal decomposition and contradict the deep-module
intent. Tests import them qualified (`using ForceCouplingMethod: spread_forces!`).

`ForceCouplingMethod.calibrate` is declared `public` but not exported: it is a cold-path tool
people call once, by its full name, and it sits on top of `FFCMConfig` and `mobility!`
without reaching into the pipeline.

**Matrix-free `mul!` interface.** `config` plus `mobility!` together
implement (or trivially adapt to via a thin operator wrapper that closes
over `config` and `Y`) the `LinearAlgebra` operator interface:

```julia
mul!(V, M, F)               # 3-arg
mul!(V, M, F, α, β)         # 5-arg: V .= α*M*F + β*V
```

where `M` is the matrix-free mobility operator backed by `config`. This
lets the operator drop straight into `IterativeSolvers.gmres!`,
`KrylovKit.linsolve`, or any solver that takes a `mul!`-compatible
linear map, without a separate adapter at the call site.

## 4. Performance discipline

- Default to `@inbounds` only where bounds have been proved safe by a
  prior assertion or explicit loop range.
- Add `@simd` and `@inline` only after a benchmark shows the bottleneck.
- For CUDA, hand-write `@cuda` kernels. Specific kernel-level choices
  (shared-memory tiling, atomics policy, register pressure) are
  benchmark-driven and live in
  [docs/src/devdocs/gpu-architecture.md](docs/src/devdocs/gpu-architecture.md)
  as they are validated. The only universal CUDA rule is the boundary one
  (§7): CUDA appears in `ext/`, never in `src/`.
- The benchmark suite in `benchmark/` is the regression detector. Run it
  before claiming a performance improvement. It times what a plain `julia`
  run gives users: no system-level tweaks (socket binding, memory policies,
  clock locks); effects of the machine are recorded as caveats with the
  results.
- **Type stability is strict.** No abstract types in struct fields; no
  non-`const` globals captured by hot-path functions. Every public
  hot-path function must pass `Test.@inferred` in tests; whole-module
  dispatch and inference health is audited with **JET.jl** (a test dependency). Allocation count (§2 rule 4) and inference
  (this rule) are two separate guarantees — both must be pinned.
- **Data layout.** Use `StructArrays.jl` for collections of physical
  entities — preserves per-particle readability (`particles[i]` returns
  the typed unit) while giving Struct-of-Arrays memory layout for
  SIMD-over-particles. Flat-vector adapters live at LinAlg API
  boundaries (e.g. `mul!`). Specific layout choices and the cost of any
  AoS↔SoA shuffles belong in the Developer documentation
  (`docs/src/devdocs/`).
- **SciML performance practices**
  (<https://github.com/SciML/SciMLStyle>) apply where this section does
  not already cover them: function barriers around unavoidable type
  instability, `let`-block rebinding to avoid boxed closure captures,
  and no splatting of long argument lists in hot code. This import is
  performance-only — syntax, formatting, and docstrings stay Blue per
  §8; do not adopt SciML aesthetic or docstring conventions.

## 5. Tests

**Order of operations for any new public API** (TDD):

1. Contract first: write the docstring of the new or changed public symbol
   — its job, `# Arguments`, `# Returns`, and preconditions — and, for a
   new algorithmic component, a stub section on the relevant docs page.
   The contract is written before the implementation exists.
2. A **failing** test pinning the behavior — typically an analytical case
   (single-sphere Stokes drag, periodic two-sphere pair, a known
   reference sum) asserted at `sqrt(eps(T))` tolerance.
3. Implementation, until the test goes green.

The test must exist *before* the code, not alongside it. A test added
after the implementation can only confirm what the code does; the
failing-test-first discipline forces the contract to be written down
before the implementation biases it.

**Test taxonomy** (three buckets under `test/`; `runtests.jl` and the
shared fixtures in `test_utilities.jl` stay at the top level):

- `accuracy/` — paper-derived or analytical correctness tests at the
  documented tolerance, one file per pipeline step plus the assembled
  operator.
- `api/` — hot-path boundary checks: allocation count via
  `BenchmarkTools.@ballocated`, type stability via `Test.@inferred`.
- `hygiene/` — whole-package audits: **Aqua.jl** (method ambiguities,
  unbound type parameters, stale `[deps]`), **JET.jl** call-graph
  inference, and the exported-surface check.
- `cuda/` — CPU↔CUDA parity tests; only run when a functional CUDA
  device is available.

`Aqua.jl` audits package hygiene; **JET.jl** audits dispatch and
inference. Both are test dependencies.

## 6. Reference implementation (dev-only)

The C++/CUDA reference this package was validated against is the paper
authors' racksa/cuFCM (<https://github.com/racksa/cuFCM>). The port is
complete and parity-validated; a gitignored clone may sit at `cuFCM/` for
dev-only benchmarking experiments. Do not cite it, its identifiers, or
its file names from shipped artefacts. The tracked record of its measured
performance is `benchmark/cufcm-baseline.md`, and the sanctioned
acknowledgments are one line each in `README.md` and
`docs/src/devdocs/gpu-architecture.md`.

## 7. Don'ts

- Don't introduce abstractions for hypothetical second use cases.
- Don't half-implement: a function either does its documented job and
  has a passing test, or it `error("not yet implemented")`s.
- Don't import CUDA from `src/`. CUDA only appears in `ext/`.
- Don't add a new public API symbol without its documented contract
  (docstring, plus a docs-page section where warranted) and a failing
  test that drove the implementation (§5).
- Don't generate docs that claim functionality that isn't tested.
- Don't reference CLAUDE.md or any other agent/process file (plan files,
  changelogs, local instructions) from `src/`, `test/`, or `docs/`.
  Shipped artefacts cite the paper or the package docs; the code stands
  on its own.

## 8. Style

- Use [Blue style](https://github.com/JuliaDiff/BlueStyle): 92-character line
  limit, 4-space indent.
- When a function call or signature would exceed 92 characters, wrap it using
  the **first** of these stages that fits. Do not skip stages.

  **Stage 0** — fits under 92 characters; do not wrap:
```julia
  func(arg1, arg2; kw1 = 1, kw2 = 2)
```

  **Stage 1** — all arguments on one wrapped line:
```julia
  func(
      arg1, arg2; kw1 = 1, kw2 = 2,
  )
```

  **Stage 2** — positional and keyword arguments on separate lines:
```julia
  func(
      arg1, arg2;
      kw1 = 1, kw2 = 2,
  )
```

  **Stage 3** — one argument per line:
```julia
  func(
      arg1,
      arg2;
      kw1 = 1,
      kw2 = 2,
  )
```

  The opening parenthesis stays on the original line. The closing parenthesis
  sits on its own line at the call's indent. The same staging applies to
  function definitions.

- Tuning values (step sizes, margins, time budgets) used by one function are
  keyword defaults of that function, each explained in its docstring's
  `# Keywords` section, not module-level `const`s. A value several functions
  share gets one owner, which the others ask.
