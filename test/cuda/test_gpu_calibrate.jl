using Test
using ForceCouplingMethod
using ForceCouplingMethod:
    CalibrationProblem,
    _gpu_grid_bytes,
    _grid_buffer_bytes,
    _random_forces,
    _reclaim_gpu_memory,
    _relative_velocity_error,
    _smallest_accurate_ratios,
    calibrate
using CUDA
using Random: Xoshiro

# Calibration on the GPU (paper §5): the very accurate answer computed there agrees with the
# processor's, the accuracy searches find the same values, the settings found meet the
# tolerance against standard FCM computed independently on the processor, and the memory a
# calibration counts on is really there. Runs only where the kernels load (guarded in
# runtests.jl). The suspensions are small, so that each calibration takes under a minute.

# The GPU memory a configuration takes once it has made a call: its buffers, its FFT plans
# and their work areas. A function of its own, so that the configuration can be freed once
# it returns.
function _gpu_memory_in_use(::Type{T}, N, M) where {T}
    L = (T(M), T(M), T(M))
    Y = T(M) .* rand(Xoshiro(2024), T, 3, N)
    F = _random_forces(T, N)
    free_before = _reclaim_gpu_memory()
    config = FFCMConfig{T}(;
        L, R_c = T(4), N, kernel_widths_ratio = T(2), num_grid_points = Int32.((M, M, M)),
        M_G = 8, viscosity = T(1), gpu_acceleration = true,
    )
    mobility!(similar(Y), config, Y, F)
    CUDA.synchronize()
    return free_before - CUDA.free_memory()
end

@testset "A grid is proposed for the GPU only where it has room for all it takes" begin
    # A first configuration loads the FFT library and the kernels, which stay loaded.
    _gpu_memory_in_use(Float32, 100, 32)
    for T in (Float32, Float64)
        N, M = 1000, 256
        used = _gpu_memory_in_use(T, N, M)
        # With only this much free, calibration would allow no larger a grid than this one.
        @test _gpu_grid_bytes(T, N, used) ≤ _grid_buffer_bytes(T, (M, M, M))
        # Once the configuration is no longer used, its memory goes back to the GPU.
        free_before = _reclaim_gpu_memory()
        _gpu_memory_in_use(T, N, M)
        @test _reclaim_gpu_memory() ≥ free_before - used / 100
    end
end

const _GPU_CALIBRATION_L = (40.0, 40.0, 40.0)
const _GPU_CALIBRATION_Y = _non_overlapping_suspension(_GPU_CALIBRATION_L, 1222)
const _GPU_CALIBRATION_F = _random_forces(Float64, 1222)

@testset "The right answer computed on the GPU agrees with the processor's" begin
    planning = (; fft_planning = :estimate, fft_threads = 1)
    Y, F, L = _GPU_CALIBRATION_Y, _GPU_CALIBRATION_F, _GPU_CALIBRATION_L
    on_processor = CalibrationProblem(Y, F, L, 1e-3; planning...)
    on_gpu = CalibrationProblem(Y, F, L, 1e-3; planning..., gpu_acceleration = true)
    @test on_gpu.reference_on_gpu
    @test on_gpu.measuring_width == on_processor.measuring_width
    # Both are in double precision, so they differ only by rounding.
    @test _relative_velocity_error(on_gpu.exact_velocities, on_processor.exact_velocities) ≤
          sqrt(eps(Float64))
end

@testset "The accuracy searches agree between the processor and the GPU (§5.1, §5.2)" begin
    planning = (; fft_planning = :estimate, fft_threads = 1)
    for T in (Float32, Float64)
        Y, F, L = T.(_GPU_CALIBRATION_Y), T.(_GPU_CALIBRATION_F), T.(_GPU_CALIBRATION_L)
        found = map((false, true)) do gpu_acceleration
            problem = CalibrationProblem(Y, F, L, 1e-3; planning..., gpu_acceleration)
            return _smallest_accurate_ratios(problem, 1e-3)
        end
        on_processor, on_gpu = found
        # The errors differ only by rounding, which can still move a search by one step
        # (0.02 in Σ/Δx, 1 in M_G, 0.1 in λ_ε) where an error sits right at its target.
        @test abs(on_gpu.kernel_resolution - on_processor.kernel_resolution) ≤ 0.025
        @test abs(on_gpu.M_G - on_processor.M_G) ≤ 1
        @test abs(on_gpu.cutoff_ratio - on_processor.cutoff_ratio) ≤ 0.15
    end
end

@testset "Calibrating on the GPU meets the tolerance in either precision" begin
    # The check is standard FCM computed independently on the processor. Unlike the
    # processor's test, this one doesn't ask that the error stay above a thirtieth of the
    # tolerance: a call this small takes about as long on any grid on a GPU, so a finer and
    # more accurate grid can fairly be the fastest. The accuracy searches themselves are
    # held to the processor's above.
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    for (T, tolerance) in ((Float32, 1e-3), (Float64, 1e-4))
        L_T, Y_T, F_T = T.(L), T.(Y), T.(F)
        settings = @test_logs (:info, r"on the GPU") match_mode = :any calibrate(
            Y_T; L = L_T, relative_tolerance = tolerance, F = F_T, gpu_acceleration = true,
        )
        config = FFCMConfig(;
            L = L_T, N = 150, viscosity = one(T), gpu_acceleration = true, settings...,
        )
        velocity_error = _mean_relative_error(
            mobility!(similar(Y_T), config, Y_T, F_T), standard_fcm,
        )
        @test velocity_error ≤ tolerance
    end
end
