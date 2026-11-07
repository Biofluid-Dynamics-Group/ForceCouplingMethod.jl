using Test
using ForceCouplingMethod
using ForceCouplingMethod:
    CalibrationProblem, _calibration_report, _random_forces, calibrate

# Calibration is cold-path code, so this file pins its contract rather than allocations or
# inference: what it takes, what it returns, what it logs, and what it refuses. The problems
# are small, so that each calibration takes seconds.

const _API_BOX = (16.0, 16.0, 16.0)
const _API_POSITIONS = _non_overlapping_suspension(_API_BOX, 20)

_quiet_calibrate(Y; keywords...) =
    @test_logs (:info, r"Σ/Δx") match_mode = :any calibrate(Y; keywords...)

# The message of the `ArgumentError` a request is refused with.
function _refusal_message(request)
    try
        request()
    catch exception
        exception isa ArgumentError || rethrow()
        return exception.msg
    end
    return "accepted"
end

@testset "Calibration returns what FFCMConfig takes, in the caller's precision" begin
    for T in (Float32, Float64)
        L = T.(_API_BOX)
        Y = T.(_API_POSITIONS)
        settings = _quiet_calibrate(
            Y; L, relative_tolerance = 1e-3, fft_planning = :estimate,
        )
        @test keys(settings) == (:kernel_widths_ratio, :num_grid_points, :M_G, :R_c)
        @test settings.kernel_widths_ratio isa T
        @test settings.num_grid_points isa NTuple{3, Int32}
        @test settings.M_G isa Int
        @test settings.R_c isa T
        config = FFCMConfig(; L, N = size(Y, 2), viscosity = one(T), settings...)
        @test config isa FFCMConfig{T}
    end
end

@testset "The tolerance and the forces have defaults" begin
    settings = _quiet_calibrate(_API_POSITIONS; L = _API_BOX, fft_planning = :estimate)
    @test settings.M_G ≥ 2
end

@testset "Calibration leaves the positions and forces untouched" begin
    Y = copy(_API_POSITIONS)
    F = _random_forces(Float64, size(Y, 2))
    F_before = copy(F)
    _quiet_calibrate(
        Y; L = _API_BOX, F, relative_tolerance = 1e-3, fft_planning = :estimate,
    )
    @test Y == _API_POSITIONS
    @test F == F_before
end

@testset "The report gives the settings in the paper's notation" begin
    problem = CalibrationProblem(
        _API_POSITIONS, _random_forces(Float64, size(_API_POSITIONS, 2)), _API_BOX, 1e-3;
        fft_planning = :estimate, fft_threads = 1,
    )
    grid = (Int32(16), Int32(16), Int32(16))
    settings = (; kernel_widths_ratio = 2.0, num_grid_points = grid, M_G = 9, R_c = 5.0)
    record = (; measured_error = 3.0e-4, seconds = 0.002)
    report = _calibration_report(problem, (; settings, tried = [settings => record]), 1e-3)
    expected_in_report = (
        "Σ/Δx", "M_G = 9", "λ_ε", "Σ/σ = 2", "R_c/σ", "16×16×16", "0.0003", "2.0 ms",
    )
    for expected in expected_in_report
        @test occursin(expected, report)
    end

    standard = (; settings..., kernel_widths_ratio = 1.0)
    report = _calibration_report(
        problem, (; settings = standard, tried = [standard => record]), 1e-3,
    )
    @test occursin("standard FCM", report)

    missed = (; record..., seconds = Inf)
    other = (; settings..., num_grid_points = (Int32(8), Int32(8), Int32(8)))
    report = _calibration_report(
        problem, (; settings, tried = [other => missed, settings => record]), 1e-3,
    )
    @test occursin("8×8×8", report)
    @test occursin("missed", report)

    fallback = (; settings = problem.reference_settings, tried = [])
    @test occursin("reference", _calibration_report(problem, fallback, 1e-3))
end

@testset "The report says where its times were measured" begin
    planning = (; fft_planning = :estimate, fft_threads = 1)
    F = _random_forces(Float64, size(_API_POSITIONS, 2))
    grid = (Int32(16), Int32(16), Int32(16))
    settings = (; kernel_widths_ratio = 2.0, num_grid_points = grid, M_G = 9, R_c = 5.0)
    record = (; measured_error = 3.0e-4, seconds = 0.002)
    outcome = (; settings, tried = [settings => record])

    on_processor = CalibrationProblem(_API_POSITIONS, F, _API_BOX, 1e-3; planning...)
    @test occursin("on the processor", _calibration_report(on_processor, outcome, 1e-3))

    # So little room for the very accurate calculation that it stays on the processor, so
    # that this needs no GPU.
    on_gpu = CalibrationProblem(
        _API_POSITIONS, F, _API_BOX, 1e-3;
        planning..., gpu_acceleration = true, free_gpu_bytes = 250_000,
    )
    @test occursin("on the GPU", _calibration_report(on_gpu, outcome, 1e-3))
end

@testset "The progress message rounds the settings found, like the report" begin
    # On this problem the searches end on values held as 0.8400000000000001 and
    # 3.4000000000000004.
    progress = r"Σ/Δx = \d\.\d{1,2} and λ_ε = \d+\.\d; now timing grids"
    @test_logs (:info, progress) match_mode = :any calibrate(
        _API_POSITIONS; L = _API_BOX, relative_tolerance = 1e-2, fft_planning = :estimate,
    )
end

@testset "Bad requests are refused at once, saying what is wrong" begin
    Y, L = _API_POSITIONS, _API_BOX
    N = size(Y, 2)
    @test occursin("3xN", _refusal_message(() -> calibrate(zeros(4, 3); L)))
    @test occursin("3xN", _refusal_message(() -> calibrate(zeros(3, 0); L)))
    @test occursin("F", _refusal_message(() -> calibrate(Y; L, F = zeros(3, N + 1))))
    @test occursin("zero", _refusal_message(() -> calibrate(Y; L, F = zeros(3, N))))

    not_finite = copy(Y)
    not_finite[2, 3] = NaN
    @test occursin("finite", _refusal_message(() -> calibrate(not_finite; L)))
    forces_not_finite = _random_forces(Float64, N)
    forces_not_finite[1, 1] = Inf
    @test occursin("finite", _refusal_message(() -> calibrate(Y; L, F = forces_not_finite)))

    negative_side = (16.0, -16.0, 16.0)
    @test occursin("positive", _refusal_message(() -> calibrate(Y; L = negative_side)))
    for tolerance in (1e-1, 0.0, -1e-3, NaN, 1e-11)
        message = _refusal_message(() -> calibrate(Y; L, relative_tolerance = tolerance))
        @test occursin("relative_tolerance", message)
    end
    # Rounding hides the error sooner in single precision.
    message = _refusal_message(
        () -> calibrate(Float32.(Y); L = Float32.(L), relative_tolerance = 1e-5),
    )
    @test occursin("relative_tolerance", message)

    message = _refusal_message(() -> calibrate(Y; L, fft_planning = :x))
    @test occursin("fft_planning", message)
    message = _refusal_message(() -> calibrate(Y; L, fft_threads = 0))
    @test occursin("fft_threads", message)

    message = _refusal_message(() -> calibrate(Float16.(Y); L = Float16.(L)))
    @test occursin("Float32", message)
end

@testset "A box too small to hold the Gaussian is refused, sooner when tighter" begin
    Y = [2.0; 2.0; 2.0;;]
    small_box = (5.0, 5.0, 5.0)
    @test occursin("box", _refusal_message(() -> calibrate(Y; L = small_box)))
    box = (6.0, 6.0, 6.0)
    @test occursin("box", _refusal_message(() -> calibrate(Y; L = box)))
    settings = _quiet_calibrate(
        Y; L = box, relative_tolerance = 1e-2, fft_planning = :estimate,
    )
    @test settings.M_G ≥ 2
end

@testset "A box with no room for the very accurate calculation is refused" begin
    # This box holds the Gaussian, but at this tolerance it calls for a 16³ grid, and the
    # generous stencil rounds up to 17 points.
    Y = [3.0; 3.0; 3.0;;]
    message = _refusal_message(
        () -> calibrate(Y; L = (6.9, 6.9, 6.9), relative_tolerance = 2e-5),
    )
    @test occursin("no room", message)
end

@testset "A box whose sides share no usable spacing is refused" begin
    L = (16.0, 16.0 * sqrt(2), 16.0)
    message = _refusal_message(() -> calibrate(_API_POSITIONS; L))
    @test occursin("equal spacing", message)
end

@testset "Two particles at the same position are refused, saying so" begin
    Y = [5.0 5.0; 5.0 5.0; 5.0 5.0]
    @test occursin("position", _refusal_message(() -> calibrate(Y; L = _API_BOX)))
end

@testset "A GPU with no room for any grid is refused, saying so" begin
    F = _random_forces(Float64, size(_API_POSITIONS, 2))
    message = _refusal_message(
        () -> CalibrationProblem(
            _API_POSITIONS, F, _API_BOX, 1e-3;
            fft_planning = :estimate,
            fft_threads = 1,
            gpu_acceleration = true,
            free_gpu_bytes = 0,
        ),
    )
    @test occursin("memory", message)
end
