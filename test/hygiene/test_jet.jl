using Test
using JET
using ForceCouplingMethod
using ForceCouplingMethod: assign_cells!, wrap_positions!, sort_particles_by_cell!,
    spread_forces!, stokes_solve!, interpolate_velocities!,
    correct_velocities!, mobility!, FFCMMobility
using LinearAlgebra: mul!

# JET catches dispatch / inference problems that `@inferred` alone can miss
# (e.g., method-instability in callees, abstract field accesses) by walking
# the full inferred call graph from each entry point. `@test_call` fails on
# inference *errors* anywhere in the graph; `@test_opt` additionally fails on
# any runtime dispatch surviving in optimized code — the stronger guarantee
# the allocation-free hot path relies on. `target_modules = (ForceCouplingMethod,)` keeps
# the optimization audit scoped to this package's code (FFTW internals are
# not ours to fix).

function _jet_test_inputs(::Type{T}) where {T}
    N = 16
    L = (T(4), T(6), T(8))
    config = FFCMConfig{T}(;
        L = L, R_c = T(1), N = N, _fcm_grid_kwargs(L)...,
    )
    Y = T[
        L[i] * (T(0.05) + T(0.9) * T(n - 1) / T(N - 1)) for i in 1:3, n in 1:N
    ]
    F = T[T(0.1) * n * (i - 2) for i in 1:3, n in 1:N]
    return config, Y, F, N
end

@testset "JET: inference health for hot-path entry points" begin
    for T in (Float32, Float64)
        config, Y, F, N = _jet_test_inputs(T)
        @test_call wrap_positions!(Y, config.L)
        @test_call assign_cells!(config, Y)
        @test_call sort_particles_by_cell!(config, Y, F)
        @test_call spread_forces!(config)
        @test_call stokes_solve!(config)
        V = zeros(T, 3, N)
        @test_call interpolate_velocities!(V, config)
        @test_call correct_velocities!(V, config)
        @test_call mobility!(V, config, Y, F)
        M = FFCMMobility(config, Y)
        f = vec(F)
        v = zeros(T, 3N)
        @test_call mul!(v, M, f)
        @test_call mul!(v, M, f, T(2), T(0.5))
        @test_call M * F
    end
end

@testset "JET: cold-path constructor call graph is sound" begin
    # The constructor may allocate and branch dynamically (it is the cold
    # path), but its call graph must still be free of inference errors.
    for T in (Float32, Float64)
        L = (T(4), T(6), T(8))
        @test_call FFCMConfig{T}(;
            L = L, R_c = T(1), N = 16, _fcm_grid_kwargs(L)...,
        )
        @test_call FFCMConfig(;
            L = L, R_c = T(1), N = 16, _fcm_grid_kwargs(L)...,
        )
    end
end

@testset "JET: calibration's call graph is sound" begin
    # Calibration is cold-path code and may dispatch dynamically, but its call graph must
    # be free of inference errors in either precision. The configuration's backend is chosen
    # at run time, so the graph reaches the GPU compiler when CUDA is loaded, and
    # `target_modules` keeps the audit to this package's code.
    for T in (Float32, Float64)
        L = (T(24), T(24), T(24))
        Y = zeros(T, 3, 16)
        F = ones(T, 3, 16)
        @test_call target_modules = (ForceCouplingMethod,) (
            ForceCouplingMethod.CalibrationProblem(
                Y, F, L, 1e-3; fft_planning = :estimate, fft_threads = 1,
            )
        )
        @test_call target_modules = (ForceCouplingMethod,) (
            ForceCouplingMethod.calibrate(Y; L, relative_tolerance = 1e-3)
        )
    end
end

@testset "JET: hot path is free of runtime dispatch" begin
    for T in (Float32, Float64)
        config, Y, F, N = _jet_test_inputs(T)
        V = zeros(T, 3, N)
        M = FFCMMobility(config, Y)
        f = vec(F)
        v = zeros(T, 3N)
        @test_opt target_modules = (ForceCouplingMethod,) mobility!(V, config, Y, F)
        @test_opt target_modules = (ForceCouplingMethod,) mul!(v, M, f)
        @test_opt target_modules = (ForceCouplingMethod,) mul!(v, M, f, T(2), T(0.5))
    end
end
