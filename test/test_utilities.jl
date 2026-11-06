# Shared helpers for the test suite. Included once by runtests.jl before any
# test file, so every helper here is visible to all tests.

using ForceCouplingMethod
using LinearAlgebra: norm
using Random: Xoshiro

# Absolute tolerance for quantities that should be ≈ 0, where a relative
# tolerance is ill-defined: atol ≈ rtol/100 with rtol = sqrt(eps(T)).
_near_zero_atol(::Type{Float32}) = 1.0f-6
_near_zero_atol(::Type{Float64}) = 1.0e-10

# Picks `Σ_over_σ`, `num_grid_points`, and `M_G` for tests that only exercise
# the step-1 / step-2 cell-list machinery and do not care about the FCM grid.
# `num_grid_points` is sized so h is isotropic across axes (paper §3
# assumption); a target h of `L[1] / 8` works for every L the existing tests
# use.
function _fcm_grid_kwargs(L::NTuple{3, T}) where {T}
    h_target = L[1] / 8
    num_grid_points = ntuple(i -> Int32(round(Int, L[i] / h_target)), 3)
    return (
        kernel_widths_ratio = T(2),
        num_grid_points = num_grid_points,
        # The stencil cannot exceed the grid on any axis; clamp for thin domains
        # where a short axis yields fewer than 8 grid points.
        M_G = min(8, Int(minimum(num_grid_points))),
        viscosity = T(1),
    )
end

# A small multi-particle configuration shared by the step and operator tests.
# Grid and kernel widths follow the regime the accuracy tests use (h isotropic,
# M_G < M); override any keyword for a test that needs a different geometry.
# Tests pin the instant heuristic planner: the measured planning of the package
# default buys per-call transform speed the suite does not need, and the
# planner-effort contract has its own test (test_fft_planning.jl).
function _standard_test_config(
    ::Type{T};
    N,
    L = (T(8), T(8), T(8)),
    num_grid_points = (Int32(16), Int32(16), Int32(16)),
    M_G = 8,
    Σ_over_σ = T(2),
    μ = T(1),
    R_c = T(1),
    fft_planning = :estimate,
    fft_threads = 1,
    gpu_acceleration = false,
) where {T}
    return FFCMConfig{T}(;
        L = L, R_c = R_c, N = N, kernel_widths_ratio = Σ_over_σ, viscosity = μ,
        num_grid_points = num_grid_points, M_G = M_G, fft_planning = fft_planning,
        fft_threads = fft_threads, gpu_acceleration = gpu_acceleration,
    )
end

# GPU-backed counterpart of `_standard_test_config` for the CPU↔CUDA parity
# tests. Only constructible where the CUDA extension is loaded and a device is
# functional; the per-step parity tests (added with each GPU kernel) call it
# under a `CUDA.functional()` guard.
_gpu_config(::Type{T}; kwargs...) where {T} =
    _standard_test_config(T; gpu_acceleration = true, kwargs...)

# Compares a CPU result against a GPU result by moving both to the host before
# `isapprox`. Tolerances follow the suite convention: relative `sqrt(eps(T))`
# with the near-zero absolute floor from `_near_zero_atol`.
function _cpu_gpu_isapprox(cpu, gpu; rtol, atol)
    return isapprox(Array(cpu), Array(gpu); rtol = rtol, atol = atol)
end

# Probe whether this device can load the FFCM device kernels. Some partially
# supported toolkit/device combinations compile the kernels to PTX but reject the
# module at load with ERROR_NOT_SUPPORTED — notably the float-to-int conversion
# intrinsic (round/floor to Int32) on CUDA 12.9 with a Pascal (sm_61) card. The
# kernel-dependent parity tests run only where this canary loads, so the suite stays
# green on such a host while still exercising the GPU path wherever it is supported.
_canary_int_conversion_kernel!(a) =
    (@inbounds a[1] = Float32(round(Int32, a[1])); nothing)

function _gpu_kernels_loadable()
    try
        a = CUDA.ones(Float32, 1)
        CUDA.@cuda threads = 1 _canary_int_conversion_kernel!(a)
        CUDA.synchronize()
        return true
    catch
        return false
    end
end

# Positions and forces for a compact particle cluster around the box centre:
# spacing 0.25 on a 4x4x4 sub-lattice, so many pairs fall within R_c = 1 and
# the pair-correction branch is exercised. Forces are a deterministic
# non-trivial pattern.
function _clustered_cloud(::Type{T}, N) where {T}
    Y = Matrix{T}(undef, 3, N)
    F = Matrix{T}(undef, 3, N)
    for n in 1:N
        Y[1, n] = T(4) + T(0.25) * ((n - 1) % 4)
        Y[2, n] = T(4) + T(0.25) * ((n - 1) ÷ 4 % 4)
        Y[3, n] = T(4) + T(0.25) * ((n - 1) ÷ 16)
        F[1, n] = T(0.1) * n
        F[2, n] = T(-0.2) * n
        F[3, n] = T(0.3) * n
    end
    return Y, F
end

# Reciprocal-lattice sum of the σ-regularised periodic Stokeslet self-mobility
# (paper §3 equations (32)–(33), Fourier form), the true M^VF self-mobility a
# single particle feels from its own periodic images. Independent of Σ. Shared
# closed-form reference of the single-sphere end-to-end tests
# (test_single_sphere_mobility.jl, test_fft_planning.jl).
function _periodic_self_mobility_xx(σ::T, μ::T, Lx::T; n_max::Int = 15) where {T}
    s = zero(T)
    for nx in -n_max:n_max, ny in -n_max:n_max, nz in -n_max:n_max
        (nx == 0 && ny == 0 && nz == 0) && continue
        kx = T(2π * nx / Lx)
        ky = T(2π * ny / Lx)
        kz = T(2π * nz / Lx)
        k² = kx * kx + ky * ky + kz * kz
        s += exp(-σ^2 * k²) / (μ * k²) * (one(T) - kx * kx / k²)
    end
    return s / Lx^3
end

# Positions of `N` unit spheres placed at random in a periodic box, none overlapping
# another, like the suspensions of the paper's §5.2. Each sphere is kept only if no earlier
# one lies closer than a diameter. Every new sphere is compared with all the others, so this
# suits small problems only.
function _non_overlapping_suspension(L::NTuple{3, T}, N; seed = 2024) where {T}
    rng = Xoshiro(seed)
    Y = Matrix{T}(undef, 3, N)
    placed = 0
    while placed < N
        candidate = ntuple(i -> L[i] * rand(rng, T), 3)
        overlaps = any(1:placed) do other
            separation = ntuple(3) do i
                d = candidate[i] - Y[i, other]
                return d - L[i] * round(d / L[i])
            end
            return sum(abs2, separation) < 4
        end
        overlaps && continue
        placed += 1
        Y[:, placed] .= candidate
    end
    return Y
end

# Standard FCM on a very fine grid, in double precision (Σ = σ, where the pair correction
# vanishes, paper §3): the independent answer a calibration's error is checked against. It
# sets the grid and stencil by hand and uses no calibration code.
function _standard_fcm_velocities(Y, F, L)
    σ = 1 / sqrt(π)
    smallest_side = minimum(L)
    M = nextprod((2, 3, 5, 7), ceil(Int, 2 * smallest_side / σ))
    num_grid_points = ntuple(i -> Int32(round(Int, M * L[i] / smallest_side)), 3)
    config = FFCMConfig{Float64}(;
        L = Float64.(L), R_c = Float64(smallest_side) / 2, N = size(Y, 2),
        kernel_widths_ratio = 1.0, viscosity = 1.0, num_grid_points,
        M_G = min(31, Int(minimum(num_grid_points))), fft_planning = :estimate,
    )
    return mobility!(Matrix{Float64}(undef, size(Y)), config, Float64.(Y), Float64.(F))
end

# The paper's error measure (§5.1, equation (69)), written out here so that checking a
# calibration's accuracy doesn't lean on the calibration's own code.
function _mean_relative_error(V, exact)
    total = sum(axes(exact, 2)) do n
        return norm(view(V, :, n) - view(exact, :, n)) / norm(view(exact, :, n))
    end
    return total / size(exact, 2)
end

# Fills the config's sorted-particle buffers directly (bypassing steps 1–2)
# with every particle at the box centre and a deterministic force pattern —
# sufficient input for spread/interpolate API checks, where positions only
# need to be in-domain.
function _fill_sorted_midbox!(config, L::NTuple{3, T}, N) where {T}
    for n in 1:N
        for i in 1:3
            config.particles.Y_sorted[i, n] = L[i] * T(0.5)
            config.particles.F_sorted[i, n] = T(0.1) * n
        end
        config.cells.original_index[n] = Int32(n)
    end
    return config
end
