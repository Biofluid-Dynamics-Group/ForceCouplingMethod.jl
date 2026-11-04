# GPU architecture

This page documents how the CUDA backend is organised: the extension boundary, the
device kernel architecture of each pipeline step, the host↔device staging that hides
device arrays from callers, and the parity contract binding the two backends. The
user-facing story is on the [GPU acceleration](../gpu.md) page.

The GPU backend's kernel architecture follows the paper authors' reference CUDA
implementation, [cuFCM](https://github.com/racksa/cuFCM), against which this package
was validated for numerical agreement and throughput; the measured reference baseline
is kept in `benchmark/cufcm-baseline.md` in the repository. Performance-motivated
deviations from that validated architecture are benchmark-gated, and their outcomes are
recorded on this page as they land.

## Backend selection is a construction-time type choice

The configuration owns every per-call buffer, and buffers must live where the kernels
run — host memory for the CPU backend, device memory for the GPU backend. The backend is
therefore fixed when the buffers are allocated, at construction, and recorded in the
configuration's *type* through the storage-type parameters of its
[sub-structs](@ref "Buffer ownership: the five sub-structs"). The step
functions dispatch on their buffer arguments with no runtime branch, and `mobility!` is
identical source for both backends.

The user expresses the choice as the single intent flag `gpu_acceleration::Bool` rather
than naming a device array type: the flag hides all CUDA vocabulary from callers, routes
the cold-path buffer assembly, and is not stored.

## The extension boundary

CUDA appears only in the `ext/ForceCouplingMethodCUDAExt` package extension, never in `src`. The
extension supplies the GPU buffer/plan assembly by adding a method to a function
declared in `src` whose only `src` method throws an `ArgumentError` naming the
requirement. That fallback has a useful inference consequence: when CUDA is not loaded,
the GPU branch of the constructor infers to `Union{}`, so the constructor remains fully
inferred on the default CPU path.

GPU construction launches no kernel: it allocates device buffers and creates the cuFFT
plans, neither of which JIT-compiles device code. The buffers are left unzeroed — every
buffer is written before it is read on a `mobility!` call — which keeps construction
independent of the device's kernel-JIT path (the CPU backend zeros its grids only
because FFTW's measuring planner dirties them).

## Kernel architecture by step

The device kernels are hand-written `@cuda` kernels, added as device methods of the same
function-barrier kernels the CPU defines, dispatched on the device buffer types.

**Step 1 — wrap, hash, cell list.** The wrap and hash are grid-stride kernels, one
thread per particle, running the same fold / floor / clamp / linearise arithmetic as the
CPU (the hash is integer arithmetic on the same scalars, so it is bit-identical). The
cell list is a parallel **counting sort**: an atomic-increment histogram, a prefix scan
(`accumulate!`) to the per-cell ranges, and an atomic-cursor scatter of the permutation.
The per-cell ranges depend only on the counts and match the CPU's exactly; the atomic
scatter is **not stable**, so the intra-cell order of `original_index` (and hence the
column order of the sorted buffers) is unspecified — numerically immaterial, since every
downstream sum over a cell is order-independent to round-off. The half-shell **neighbour
map** consumed by the correction (13 forward neighbours per cell) is geometry-only, so
it is built once on the host at construction and copied to the device; it is the `cells`
sub-struct's backend-divergent field.

**Step 3 — force spreading.** One block per particle (block-stride beyond the launch
grid), 32 threads per block — a single warp. The block cooperatively stages the
separable per-axis stencil — 1-D Gaussian weights, squared distances, wrapped 1-based
grid indices — in dynamic shared memory, which *replaces* the CPU's global stencil
scratch (the GPU path leaves the `stencil` sub-struct unused). The threads then sweep
the ``M_G^3`` support, each forming the modified-kernel weight
``(a_0 + a_2 r^2)\, g_x g_y g_z`` and atomically adding the monopole force contribution
to the three grid components. The polynomial coefficients and Gaussian normalisation
come from the same host-side helper the CPU kernel uses, and the anchor and periodic
wrap follow the CPU conventions, so only the atomic accumulation order differs from the
CPU result.

**Step 4 — Stokes solve.** `stokes_solve!` needs no GPU-specific entry: its forward and
inverse transforms go through the `AbstractFFTs` `mul!` interface, which dispatches to
the cuFFT plans the assembly built. The per-mode projection gets a device method: one
thread per Fourier mode over the half-spectrum, grid-stride, each thread reading
``k_x[i_x], k_y[i_y], k_z[i_z]`` from the per-axis wavenumber vectors the assembly left
resident on the device — so the two backends see bit-identical wavevectors — and
applying the same normalised projector with the ``\boldsymbol{k} = \boldsymbol{0}``
gauge fix as a guarded write of zero. The whole step stays device-resident: forward
cuFFT → in-place projection → inverse cuFFT, with no host round trip.

**Step 5 — interpolation / gather.** The gather mirror of the spread: one block per
particle, 32 threads, the same shared-memory stencil staging. Each thread accumulates a
partial kernel-weighted velocity over its slice of the ``M_G^3`` support; because the
32-thread block is a single warp, the partials are summed with a **warp-shuffle
reduction** — no shared scratch and no atomics, since the gather only reads the grid and
each particle writes its own output column. Lane 0 applies the ``h^3`` quadrature factor
once and writes `V[:, original_index[np]]`, folding the unsort into the write exactly as
the CPU kernel does.

**Step 6 — pairwise correction.** One thread per sorted particle, grid-stride. The
thread recomputes its particle's cell from the wrapped position (the same hash as step
1), seeds its accumulator with the folded self term ``\delta\boldsymbol{F}_i``, sweeps
every other particle in its own cell accumulating the correction to itself only, then
sweeps the 13 forward neighbour cells from the device neighbour map: each in-range pair
is visited once, and because the correction tensor is symmetric, the same pair scalars
give both the correction to ``i`` (kept in registers) and the correction to ``j``
(atomically added to ``j``'s original-order column). The accumulated self-plus-own
correction is atomically added to `V[:, original_index[i]]`. The pair scalars and the
minimum image are the CPU's own helpers — the minimum image is exactly antisymmetric, so
each pair's scalars are bit-identical across backends and only the atomic summation
order differs.

This step's dispatch seam is wider than the others: the shared kernel signature carries
the neighbour map and the inverse cell size, and dispatch is on the neighbour-map
argument — `nothing` selects the CPU method (an on-the-fly 27-cell gather that ignores
both extra arguments), a device vector selects the GPU method.

## The host↔device boundary

The caller of `mobility!` passes ordinary host ``3 \times N`` arrays for both backends.
The assembled operator stages them across the boundary: `mobility!` opens with a stage
call and closes with a retrieve call, both dispatched on the `particles` sub-struct
storage type. On the CPU backend the stage returns the caller's arrays unchanged and the
retrieve is a no-op — byte-for-byte the CPU-only behaviour, copy-free. On the GPU
backend the stage `copyto!`s `Y` and `F` into device staging buffers and the retrieve
`copyto!`s the device velocities back; everything between the two edges stays resident
on the device. Because the seam dispatches like the step kernels, `mobility!` remains
one source with no backend test in its body, and the `FFCMMobility` operator keeps its
host scratch and drives a GPU configuration unchanged — the traffic is hidden one level
below the linear-operator interface.

The GPU-backed `particles` sub-struct owns three device buffers for this (position and
force uploads, velocity download); the CPU-backed one carries `nothing` in their place.

## The parity contract

The GPU backend is correct when, for the same positions and forces, it reproduces the
CPU backend's velocities to the documented tolerance (``\sqrt{\mathrm{eps}(T)}``
relative, with the absolute floor for near-zero entries). Parity is asserted per
pipeline step and end to end by `test/cuda/`, which runs wherever `CUDA.functional()`
is true; each per-step test writes the step's inputs directly and identically on both
backends, isolating it from the steps before it. Parity is to round-off rather than
bit-exact because the atomic scatters and parallel reductions fix the result but not the
floating-point summation order. The GPU hot path is additionally held to
`CUDA.@allocated == 0`, mirroring the CPU allocation contract.

## Deferred, benchmark-gated variants

The initial port takes the validated reference architecture unchanged; each variant
below is adopted only on a measured win against it, and outcomes are recorded here.

- A stable, deterministic ``\mathcal{O}(N)`` radix sort in place of the atomic counting
  sort (restores intra-cell determinism; relevant if reproducibility is required or
  atomic contention bottlenecks).
- An allocation-free prefix scan for large cell counts (`accumulate!` allocates a
  transient aggregate buffer on its multi-block path).
- Building the correction's neighbour shell on the fly instead of the precomputed map
  (would drop the backend-divergent `cells` field).
- Launch-shape tuning: 32-versus-larger thread blocks, grid-stride mappings, and
  no-shared-memory or thread-per-particle spread/gather variants.
- A shared spread↔gather device stencil-fill helper (the precompute is currently
  duplicated in the two kernels; the CPU shares one routine), plus naming alignment of
  the two kernels' launch helpers.
- A 27-cell gather correction (no atomics, like the CPU) versus the half-shell atomic
  dual write; a separate self-term kernel versus the folded self term.
- Caching the position upload for operators with fixed positions across a solve
  (`FFCMMobility` re-uploads `Y` every `mul!`); a zero-copy fast path when the caller
  already holds device arrays; staging-buffer minimisation.

Measured so far: on an RTX 2080 Ti, on the paper's five §5.2 problems, with positions and
forces already on the GPU. Both codes were timed over the same whole application,
including the reference implementation's work outside its own step timers.

- **`Float32`:** the port took 0.72–0.87 of the reference's time. Most of that lead comes
  from the reference's pair correction, which does part of its arithmetic in double
  precision. Without that, an estimate puts the port level on the smallest problem and
  6–17 % slower on the other four, mostly in the FFT step and interpolation.
- **`Float64`:** the port took 0.81–0.92 of the reference's time on the three larger
  problems, and 1.02 on the smallest.

An earlier single-problem figure, 111.7 % of the reference's throughput, compared the
port's fastest call with the reference's average; average against average it was about
109 %. Details are in `benchmark/cufcm-baseline.md`.
