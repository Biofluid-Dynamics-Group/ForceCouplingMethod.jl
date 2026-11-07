"""
Cell-list bookkeeping for the pairwise correction (paper §4): the per-particle
cell hash, the sort permutation, the per-cell occupancy ranges with the
counting-sort workspace, and the GPU neighbour map. The first five buffers are
integer-indexed.

The storage type `IntVector` is `Vector{Int32}` on the CPU backend and the
device vector type on the GPU backend, so the backend is a type-parameter swap.
`NeighborMap` is the single backend-divergent field: `Nothing` on the CPU (whose
correction computes neighbours on the fly) and the device half-shell map on the
GPU.

# Fields
- `cell_hash::IntVector`: per-particle 0-based linear cell index (length `N`).
- `original_index::IntVector`: sorted slot to original particle index (length `N`).
- `cell_start::IntVector`, `cell_end::IntVector`: 1-based inclusive sorted-slot
  range of each cell (length `prod(num_cells)`).
- `counting_sort_scratch::IntVector`: per-cell counting-sort workspace
  (length `prod(num_cells)`).
- `neighbor_map::NeighborMap`: `nothing` on the CPU; on the GPU the device vector
  of 13 forward half-shell neighbour cells per cell (length `13 * prod(num_cells)`),
  built by `_build_neighbor_map` and consumed by the pairwise correction.
"""
struct CellBuffers{IntVector, NeighborMap}
    cell_hash::IntVector
    original_index::IntVector
    cell_start::IntVector
    cell_end::IntVector
    counting_sort_scratch::IntVector
    neighbor_map::NeighborMap
end

"""
Particle-indexed `3xN` work buffers the cell-list pipeline folds and reorders
the caller's positions and forces into (paper §4), kept distinct from the
integer cell-list bookkeeping in `CellBuffers`.

The storage type `Mat` is `Matrix{T}` on the CPU backend and the device matrix
type on the GPU backend. `Staging` is the backend-divergent type of the
host↔device staging buffers: `Nothing` on the CPU (which consumes the caller's
host arrays directly) and the device matrix type on the GPU, where `mobility!`
uploads the caller's positions/forces into `Y_input`/`F_input` and downloads the
velocities from `V_output`, hiding the host↔device traffic.

# Fields
- `Y_wrapped::Mat`: caller positions folded into the periodic box (so the
  caller's `Y` is never mutated).
- `Y_sorted::Mat`, `F_sorted::Mat`: positions and forces in cell-sorted order.
- `Y_input::Staging`, `F_input::Staging`: `nothing` on the CPU; on the GPU the
  device-resident copies of the caller's raw `3xN` positions and forces.
- `V_output::Staging`: `nothing` on the CPU; on the GPU the device-resident `3xN`
  velocity output, copied back into the caller's `V` after the pipeline.
"""
struct ParticleBuffers{Mat, Staging}
    Y_wrapped::Mat
    Y_sorted::Mat
    F_sorted::Mat
    Y_input::Staging
    F_input::Staging
    V_output::Staging
end

"""
The real-space grid fields of the FCM solve (paper §3): the spread force
density and the interpolated fluid velocity, both `(M_x, M_y, M_z)` fields of
3-vectors.

The storage type `GridField` is a `StructArray{SVector{3, T}}` whose component
arrays are `Array{T, 3}` on the CPU backend and the device array type on the
GPU backend.

# Fields
- `force_density::GridField`: spread force density `J̃†[F]` (output of `spread_forces!`).
- `fluid_velocity::GridField`: Stokes velocity field (output of `stokes_solve!`).
"""
struct GridBuffers{GridField}
    force_density::GridField
    fluid_velocity::GridField
end

"""
Fourier-space state of the spectral Stokes solve (paper §3): the half-complex
force/velocity field, the per-axis wavevectors, and the cached FFT plans. Named
for the solve rather than the transform so a future non-spectral Stokes solver
can reuse the boundary.

The component storage swaps with the backend; the FFT plans are FFTW plans on
the CPU backend and cuFFT plans on the GPU backend.

# Fields
- `fluid_hat::SpectralField`: half-complex `(M_x÷2+1, M_y, M_z)` field reused as
  the forward-transform output and the Stokes-solved spectrum.
- `k_x::RealVector`, `k_y::RealVector`, `k_z::RealVector`: per-axis Fourier
  wavevectors in FFTW layout.
- `forward_fourier_transform::FwdTransform`,
  `inverse_fourier_transform::InvTransform`: the real-to-complex and
  complex-to-real plans.
"""
struct SolverState{SpectralField, RealVector, FwdTransform, InvTransform}
    fluid_hat::SpectralField
    k_x::RealVector
    k_y::RealVector
    k_z::RealVector
    forward_fourier_transform::FwdTransform
    inverse_fourier_transform::InvTransform
end

"""
Per-particle stencil workspace shared by spreading and interpolation (paper §5):
the separable Gaussian weights, the squared grid-point distances, and the
wrapped grid indices over the `M_G` cubic support. These are recomputed for each
particle and carry no state between `mobility!` calls.

The storage type swaps with the backend.

# Fields
- `stencil_gaussian::StencilField`: separable Gaussian stencil weights (length `M_G`).
- `stencil_r²::StencilField`: squared distances to the stencil grid points (length `M_G`).
- `stencil_index::StencilIndexField`: periodic-wrapped grid indices (length `M_G`).
"""
struct StencilBuffers{StencilField, StencilIndexField}
    stencil_gaussian::StencilField
    stencil_r²::StencilField
    stencil_index::StencilIndexField
end

"""
    FFCMConfig{T}(
        ; L, R_c, N, a = T(1), kernel_widths_ratio, num_grid_points, M_G, viscosity,
        fft_planning = :measure, fft_threads = 1, gpu_acceleration = false,
    ) -> FFCMConfig{T}
    FFCMConfig(; L, ...) -> FFCMConfig{eltype(L)}

Configuration of the Fast FCM mobility operator: the cell partition of the pairwise
correction (paper §4), the FCM grid parameters (paper §3 and §5), and the per-call buffers
sized for `N` particles. Built once and reused across `mobility!` calls, which write into
the buffers this struct owns. `N` is fixed at construction — changing `N` requires a new
`FFCMConfig`.

The compiled buffers are grouped into the `cells`, `particles`, `grid`, `solver`, and
`stencil` sub-structs; their storage types parameterize `FFCMConfig`, so selecting the GPU
backend is a type swap and the backend never appears on the `mobility!` hot path.

The type-parameter-free form infers the working precision `T` from the element type of
the domain lengths `L`, which must be a concrete subtype of `AbstractFloat`.

# Keywords
- `L::NTuple{3, T}`: the domain lengths `(L_x, L_y, L_z)` (paper §4). Each component must be
  positive.
- `R_c::T`: the cutoff radius of the pairwise correction. Must satisfy
  `0 < R_c ≤ min(L)/2`, so the minimum image is unambiguous and no particle is corrected
  against its own periodic image (paper §4).
- `N::Integer`: the number of particles; fixes the hot-path buffer sizes. Must be positive.
- `a::T = T(1)`: the particle hydrodynamic radius. Only `a == T(1)` is currently supported;
  the kernel width follows `σ = a/√π` (paper §2, the FCM radius-width relation that recovers
  the single-particle Stokes drag).
- `kernel_widths_ratio::T`: the ratio between the wider Gaussian kernel width and the
  standard FCM kernel width (paper §5). Must satisfy `kernel_widths_ratio ≥ T(1)`; ratio of
  1 is the standard FCM limit.
- `num_grid_points::NTuple{3, Int32}`: the FFT grid dimensions `(M_x, M_y, M_z)` (paper §3).
  The induced spacing `h = L_i / M_i` must be identical across axes (paper §3 assumption).
- `M_G::Integer`: the cubic stencil support per axis (paper §5). Stored as `Int32`; must
  satisfy `2 ≤ M_G ≤ min(num_grid_points)` — the stencil cannot be wider than the grid on
  any axis, or the periodic wrap would alias distinct stencil points onto the same grid
  point.
- `viscosity::T`: the fluid dynamic viscosity (paper §2, Stokes momentum balance). Must be
  positive.
- `fft_planning::Symbol = :measure`: the FFT planner effort, one of `:estimate`,
  `:measure`, or `:patient`. The measuring efforts spend construction time (seconds at
  large grids) searching for faster transforms for the repeated `stokes_solve!` calls of
  a long solve; `:estimate` plans instantly from heuristics. The computed velocities are
  independent of the effort to round-off.
- `fft_threads::Integer = 1`: the execution thread count baked into the FFT plans. Must
  be at least 1. Threaded plans execute as Julia tasks, which allocate per call: the
  allocation-free hot path requires `fft_threads = 1`. The computed velocities match the
  single-threaded result to round-off. Ignored by the GPU backend.
- `gpu_acceleration::Bool = false`: build the configuration on a CUDA GPU rather than the
  CPU. Requires the `CUDA` extension to be loaded (`using CUDA`) on a machine with a
  functional CUDA device; otherwise construction throws. `Float64` on the GPU emits a
  non-fatal warning, once in a session, since double-precision throughput is a fraction of
  single-precision on consumer NVIDIA hardware.

# Returns
- `FFCMConfig{T}`: the configuration `mobility!` operates on.

# Throws
- `ArgumentError`: if `R_c ≤ 0`, any `L_i ≤ 0`, `R_c > min(L)/2`, `N ≤ 0`, `a ≠ T(1)`
  (non-unit radius not yet implemented), `kernel_widths_ratio < 1`, `M_G < 2`,
  `M_G > min(num_grid_points)`, any `num_grid_points` component `< 1`, the grid spacing is
  anisotropic, `viscosity ≤ 0`, `fft_planning` is not one of the three planner efforts,
  `fft_threads < 1`, or `gpu_acceleration = true` without the `CUDA` extension loaded. The
  type-parameter-free form additionally throws if `eltype(L)` is not a concrete subtype of
  `AbstractFloat` (integer or mixed-precision lengths are rejected, not promoted — the
  working precision is an explicit choice).
"""
struct FFCMConfig{T <: AbstractFloat, Cells, Particles, Grid, Solver, Stencil}
    L::NTuple{3, T}
    R_c::T
    num_cells::NTuple{3, Int32}
    cell_size::NTuple{3, T}
    inv_cell_size::NTuple{3, T}
    a::T
    σ::T
    Σ::T
    num_grid_points::NTuple{3, Int32}
    M_G::Int32
    h::T
    inv_h::T
    μ::T
    cells::Cells
    particles::Particles
    grid::Grid
    solver::Solver
    stencil::Stencil
end

"""
    _wrap_wavevectors(M, L) -> Vector{T}

Returns the per-axis Fourier wavevectors `k = 2π m / L` for a full transform axis of length
`M`, in FFTW's wrap-around order: the signed mode index is `m = j - 1` on the leading
non-negative half (`j ≤ M/2 + 1`) and `m = j - 1 - M` on the trailing negative half. Used
for the `y` and `z` axes of the Stokes-solve grid; the `x` axis keeps only the non-negative
half under the transform and is built inline.

# Arguments
- `M::Int32`: the number of grid points along the axis.
- `L::T`: the box length along the axis.

# Returns
- `Vector{T}` of length `M`: the per-axis wavevector components.
"""
function _wrap_wavevectors(M::Int32, L::T) where {T}
    twoπ = T(2) * T(π)
    M_half_plus_one = M ÷ Int32(2) + Int32(1)
    return T[twoπ * (j ≤ M_half_plus_one ? T(j - 1) : T(j - 1 - M)) / L for j in 1:M]
end

"""
    _fftw_planner_flag(fft_planning) -> UInt32

Maps the public planner-effort name to the FFTW planner flag: `:estimate` to
`FFTW.ESTIMATE`, `:measure` to `FFTW.MEASURE`, `:patient` to `FFTW.PATIENT`.

# Arguments
- `fft_planning::Symbol`: one of `:estimate`, `:measure`, `:patient` (validated by the
  `FFCMConfig` constructor).

# Returns
- `UInt32`: the FFTW planner flag.
"""
function _fftw_planner_flag(fft_planning::Symbol)
    return fft_planning === :estimate ? ESTIMATE :
           fft_planning === :measure ? MEASURE : PATIENT
end

"""
    _has_equal_grid_spacing(L, num_grid_points) -> Bool

Whether the grid spacing `L_i / M_i` is the same along every axis, as the paper assumes
(§3), to within a relative tolerance of `sqrt(eps(T))`.

# Arguments
- `L::NTuple{3, T}`: the domain lengths.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`, each at least
  1.

# Returns
- `Bool`: whether the spacing is equal.
"""
function _has_equal_grid_spacing(
    L::NTuple{3, T},
    num_grid_points::NTuple{3, Int32},
) where {T}
    h_per_axis = ntuple(i -> L[i] / num_grid_points[i], 3)
    tolerance = sqrt(eps(T)) * h_per_axis[1]
    return abs(h_per_axis[2] - h_per_axis[1]) ≤ tolerance &&
           abs(h_per_axis[3] - h_per_axis[1]) ≤ tolerance
end

"""
    _cutoff_fits_box(R_c, L) -> Bool

Whether the cutoff of the pairwise correction is at most half the smallest box length, so
the minimum image is unambiguous and no particle is corrected against its own periodic
image (paper §4).

# Arguments
- `R_c::T`: the cutoff radius.
- `L::NTuple{3, T}`: the domain lengths.

# Returns
- `Bool`: whether the cutoff fits.
"""
_cutoff_fits_box(R_c::T, L::NTuple{3, T}) where {T} = R_c ≤ minimum(L) / T(2)

"""
    _support_fits_grid(M_G, num_grid_points) -> Bool

Whether the stencil is no wider than the grid on any axis. A wider one would wrap distinct
stencil points onto the same grid point (paper §5).

# Arguments
- `M_G::Integer`: the cubic stencil support per axis.
- `num_grid_points::NTuple{3, Int32}`: the grid dimensions `(M_x, M_y, M_z)`.

# Returns
- `Bool`: whether the stencil fits.
"""
_support_fits_grid(M_G::Integer, num_grid_points::NTuple{3, Int32}) =
    M_G ≤ minimum(num_grid_points)

"""
    _validate_fft_options(fft_planning, fft_threads)

Enforces the `FFCMConfig` preconditions on the FFT choices, throwing `ArgumentError` on the
first violation.

# Arguments
- `fft_planning::Symbol`: must be `:estimate`, `:measure`, or `:patient`.
- `fft_threads::Integer`: must be at least 1.
"""
function _validate_fft_options(fft_planning::Symbol, fft_threads::Integer)
    fft_planning in (:estimate, :measure, :patient) || throw(ArgumentError(
        "fft_planning must be :estimate, :measure, or :patient; got " *
        ":$(fft_planning)",
    ))
    fft_threads ≥ 1 || throw(ArgumentError(
        "fft_threads must be at least 1; got $(fft_threads)",
    ))
    return nothing
end

"""
    _validate_config_parameters(T; L, R_c, N, a, kernel_widths_ratio, num_grid_points,
        M_G, viscosity, fft_planning, fft_threads)

Enforces the `FFCMConfig` preconditions (paper §3, §4, §5), throwing `ArgumentError` on the
first violation. Backend-agnostic; runs before any buffer is allocated. The anisotropy check
needs the derived grid spacing and lives in `_derive_scalars`.

See the `FFCMConfig` docstring for the full precondition list.
"""
function _validate_config_parameters(
    ::Type{T};
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
    a::T,
    kernel_widths_ratio::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Integer,
    viscosity::T,
    fft_planning::Symbol,
    fft_threads::Integer,
) where {T <: AbstractFloat}
    R_c > zero(T) || throw(ArgumentError("R_c must be positive"))
    all(>(zero(T)), L) || throw(ArgumentError("L components must be positive"))
    _cutoff_fits_box(R_c, L) || throw(ArgumentError(
        "R_c must be at most half the smallest box length; " *
        "got R_c = $(R_c), min(L)/2 = $(minimum(L) / T(2))",
    ))
    N > 0 || throw(ArgumentError("N must be positive"))
    a == T(1) || throw(ArgumentError(
        "non-unit particle radius not yet implemented; only a = 1 is " *
        "supported, got a = $(a)",
    ))
    kernel_widths_ratio ≥ T(1) || throw(ArgumentError(
        "kernel widths ratio must be at least 1; got $(kernel_widths_ratio)",
    ))
    M_G ≥ 2 || throw(ArgumentError("M_G must be at least 2; got $(M_G)"))
    all(≥(Int32(1)), num_grid_points) ||
        throw(ArgumentError("num_grid_points components must each be ≥ 1"))
    _support_fits_grid(M_G, num_grid_points) || throw(ArgumentError(
        "M_G must not exceed the grid on any axis; got M_G = $(M_G), " *
        "min(num_grid_points) = $(minimum(num_grid_points))",
    ))
    viscosity > zero(T) || throw(ArgumentError(
        "viscosity must be positive; got $(viscosity)",
    ))
    _validate_fft_options(fft_planning, fft_threads)
    return nothing
end

"""
    _derive_scalars(T; L, R_c, a, kernel_widths_ratio, num_grid_points) -> NamedTuple

Computes the backend-independent derived scalars and lookup tables from the validated
parameters (paper §3, §4, §5): the cell-grid geometry, the kernel widths `σ`/`Σ`, the
isotropic grid spacing `h`, and the per-axis Fourier wavevectors. Throws `ArgumentError`
if the induced grid spacing is anisotropic (paper §3 assumes uniform `h`).

# Returns
- `NamedTuple` with `num_cells`, `cell_size`, `inv_cell_size`, `σ`, `Σ`, `h`, `inv_h`, and
  the host wavevectors `k_x`, `k_y`, `k_z` (the backend assembly moves them to the device).
"""
function _derive_scalars(
    ::Type{T};
    L::NTuple{3, T},
    R_c::T,
    a::T,
    kernel_widths_ratio::T,
    num_grid_points::NTuple{3, Int32},
) where {T <: AbstractFloat}
    num_cells = ntuple(i -> max(floor(Int32, L[i] / R_c), Int32(3)), 3)
    cell_size = ntuple(i -> L[i] / num_cells[i], 3)
    inv_cell_size = ntuple(i -> one(T) / cell_size[i], 3)

    σ = a / sqrt(T(π))
    Σ = kernel_widths_ratio * σ

    _has_equal_grid_spacing(L, num_grid_points) || throw(ArgumentError(
        "anisotropic grid spacing not supported (paper §3 assumes uniform h); " *
        "L_i/M_i = $(ntuple(i -> L[i] / num_grid_points[i], 3))",
    ))
    h = L[1] / num_grid_points[1]
    inv_h = one(T) / h

    M_x, M_y, M_z = num_grid_points
    fft_M_x = M_x ÷ Int32(2) + Int32(1)
    k_x = T[T(2) * T(π) * (i - 1) / L[1] for i in 1:fft_M_x]
    k_y = _wrap_wavevectors(M_y, L[2])
    k_z = _wrap_wavevectors(M_z, L[3])

    return (; num_cells, cell_size, inv_cell_size, σ, Σ, h, inv_h, k_x, k_y, k_z)
end

"""
    _grid_buffer_bytes(T, num_grid_points) -> Int

The memory the grid-sized buffers of a configuration take (paper §3): the force density and
fluid velocity, three real fields each, and the three half-complex fields of the
Fourier-space solve. These dominate the memory of a configuration on any useful grid.

# Arguments
- `T::Type{<:AbstractFloat}`: the precision.
- `num_grid_points::NTuple{3, <:Integer}`: the grid dimensions `(M_x, M_y, M_z)`.

# Returns
- `Int`: the number of bytes.
"""
function _grid_buffer_bytes(::Type{T}, num_grid_points::NTuple{3, <:Integer}) where {T}
    M_x, M_y, M_z = Int.(num_grid_points)
    real_values = 6 * M_x * M_y * M_z + 3 * 2 * (M_x ÷ 2 + 1) * M_y * M_z
    return real_values * sizeof(T)
end

"""
    _assemble_cpu_buffers(T, N, num_grid_points, M_G, num_cells_total, k_x, k_y, k_z,
        fft_planning, fft_threads) -> (cells, particles, grid, solver, stencil)

Allocates the CPU-backed sub-struct buffers and builds the FFTW plans (paper §3, §5). The
real grid buffers are allocated unzeroed and zeroed only after the plans are built on them:
the measuring planner efforts execute candidate transforms on the input array, overwriting
its contents.

# Returns
- `Tuple` of `CellBuffers`, `ParticleBuffers`, `GridBuffers`, `SolverState`,
  `StencilBuffers`, all backed by `Array`/`Vector`.
"""
function _assemble_cpu_buffers(
    ::Type{T},
    N::Integer,
    num_grid_points::NTuple{3, Int32},
    M_G::Int32,
    num_cells_total::Integer,
    k_x::Vector{T},
    k_y::Vector{T},
    k_z::Vector{T},
    fft_planning::Symbol,
    fft_threads::Integer,
) where {T}
    cells = CellBuffers(
        Vector{Int32}(undef, N),
        Vector{Int32}(undef, N),
        Vector{Int32}(undef, num_cells_total),
        Vector{Int32}(undef, num_cells_total),
        Vector{Int32}(undef, num_cells_total),
        nothing,
    )
    particles = ParticleBuffers(
        Matrix{T}(undef, 3, N), Matrix{T}(undef, 3, N), Matrix{T}(undef, 3, N),
        nothing, nothing, nothing,
    )

    M_x, M_y, M_z = num_grid_points
    fx = Array{T, 3}(undef, M_x, M_y, M_z)
    fy = Array{T, 3}(undef, M_x, M_y, M_z)
    fz = Array{T, 3}(undef, M_x, M_y, M_z)
    ux = Array{T, 3}(undef, M_x, M_y, M_z)
    uy = Array{T, 3}(undef, M_x, M_y, M_z)
    uz = Array{T, 3}(undef, M_x, M_y, M_z)
    grid = GridBuffers(
        StructArray{SVector{3, T}}((fx, fy, fz)),
        StructArray{SVector{3, T}}((ux, uy, uz)),
    )

    fft_M_x = M_x ÷ Int32(2) + Int32(1)
    fh_x = Array{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fh_y = Array{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fh_z = Array{Complex{T}, 3}(undef, fft_M_x, M_y, M_z)
    fluid_hat = StructArray{SVector{3, Complex{T}}}((fh_x, fh_y, fh_z))

    planner_flag = _fftw_planner_flag(fft_planning)
    forward_fourier_transform = plan_rfft(
        fx; flags = planner_flag, num_threads = Int(fft_threads),
    )
    inverse_fourier_transform = plan_brfft(
        fh_x, Int(M_x); flags = planner_flag, num_threads = Int(fft_threads),
    )
    solver = SolverState(
        fluid_hat, k_x, k_y, k_z,
        forward_fourier_transform, inverse_fourier_transform,
    )

    stencil = StencilBuffers(
        StructArray{SVector{3, T}}((
            Vector{T}(undef, M_G), Vector{T}(undef, M_G), Vector{T}(undef, M_G),
        )),
        StructArray{SVector{3, T}}((
            Vector{T}(undef, M_G), Vector{T}(undef, M_G), Vector{T}(undef, M_G),
        )),
        StructArray{SVector{3, Int32}}((
            Vector{Int32}(undef, M_G),
            Vector{Int32}(undef, M_G),
            Vector{Int32}(undef, M_G),
        )),
    )

    for buffer in (fx, fy, fz, ux, uy, uz)
        fill!(buffer, zero(T))
    end
    for buffer in (fh_x, fh_y, fh_z)
        fill!(buffer, zero(Complex{T}))
    end

    return (cells, particles, grid, solver, stencil)
end

"""
    _assemble_gpu_buffers(args...)

Builds the GPU-backed sub-struct buffers and cuFFT plans. The concrete method is supplied
by the `ForceCouplingMethodCUDAExt` extension; this fallback fires when
`gpu_acceleration = true` is requested without `using CUDA` having loaded the extension,
and reports that requirement.
"""
_assemble_gpu_buffers(args...) = throw(ArgumentError(
    "gpu_acceleration = true requires the CUDA backend; run `using CUDA` on a " *
    "machine with a functional CUDA device before constructing a GPU-backed FFCMConfig",
))

function FFCMConfig{T}(;
    L::NTuple{3, T},
    R_c::T,
    N::Integer,
    a::T = T(1),
    kernel_widths_ratio::T,
    num_grid_points::NTuple{3, Int32},
    M_G::Integer,
    viscosity::T,
    fft_planning::Symbol = :measure,
    fft_threads::Integer = 1,
    gpu_acceleration::Bool = false,
) where {T <: AbstractFloat}
    _validate_config_parameters(
        T;
        L, R_c, N, a, kernel_widths_ratio, num_grid_points, M_G, viscosity,
        fft_planning, fft_threads,
    )
    derived = _derive_scalars(
        T; L, R_c, a, kernel_widths_ratio, num_grid_points,
    )
    M_G_i32 = Int32(M_G)
    num_cells_total = prod(Int, derived.num_cells)

    if gpu_acceleration && T === Float64
        @warn "Float64 throughput on consumer NVIDIA GPUs is a fraction of Float32; " *
              "consider Float32 for GPU runs" maxlog = 1
    end

    cells, particles, grid, solver, stencil = if gpu_acceleration
        # The neighbour map is geometry-only; build it once on the host and let the
        # extension copy it to the device.
        _assemble_gpu_buffers(
            T, N, num_grid_points, M_G_i32, num_cells_total,
            derived.k_x, derived.k_y, derived.k_z,
            _build_neighbor_map(derived.num_cells),
        )
    else
        _assemble_cpu_buffers(
            T, N, num_grid_points, M_G_i32, num_cells_total,
            derived.k_x, derived.k_y, derived.k_z, fft_planning, fft_threads,
        )
    end

    return FFCMConfig{
        T,
        typeof(cells),
        typeof(particles),
        typeof(grid),
        typeof(solver),
        typeof(stencil),
    }(
        L,
        R_c,
        derived.num_cells,
        derived.cell_size,
        derived.inv_cell_size,
        a,
        derived.σ,
        derived.Σ,
        num_grid_points,
        M_G_i32,
        derived.h,
        derived.inv_h,
        viscosity,
        cells,
        particles,
        grid,
        solver,
        stencil,
    )
end

function FFCMConfig(; L, kwargs...)
    T = eltype(L)
    T <: AbstractFloat && isconcretetype(T) || throw(ArgumentError(
        "the element type of L must be a concrete subtype of AbstractFloat " *
        "to infer the working precision; got eltype(L) = $(T)",
    ))
    return FFCMConfig{T}(; L, kwargs...)
end
