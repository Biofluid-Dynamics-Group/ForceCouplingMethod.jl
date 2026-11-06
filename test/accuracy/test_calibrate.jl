using Test
using ForceCouplingMethod
using ForceCouplingMethod:
    CalibrationProblem,
    _candidate_settings,
    _check_reference_agreement,
    _balanced_grid_spacing,
    _fastest_candidate,
    _fastest_taking_turns,
    _fft_friendly_grids,
    _generous_ratios,
    _generous_support,
    _grid_buffer_bytes,
    _has_equal_grid_spacing,
    _is_fft_friendly,
    _measuring_width,
    _median_call_time,
    _random_forces,
    _relative_velocity_error,
    _settings_on_grid,
    _smallest_passing,
    _tighten_until_met
using StructArrays: components

# The suspensions here are small, a few hundred particles at most, so the suite stays quick.
# They test the algorithm rather than any particular suspension.

@testset "The relative velocity error matches a hand-worked example (paper eq. (69))" begin
    exact = [3.0 0.0; 4.0 2.0; 0.0 0.0]  # speeds 5 and 2
    V = [3.0 0.0; 4.0 2.0; 0.5 1.0]      # errors of size 0.5 and 1
    @test _relative_velocity_error(V, exact) ≈ (0.5 / 5 + 1 / 2) / 2
end

@testset "A particle at rest has no relative error and is left out of the average" begin
    exact = [3.0 0.0; 4.0 0.0; 0.0 0.0]
    V = [3.0 1.0; 4.0 1.0; 0.5 1.0]
    @test _relative_velocity_error(V, exact) ≈ 0.5 / 5
end

@testset "The relative error is undefined when no particle moves" begin
    @test_throws ArgumentError _relative_velocity_error(ones(3, 2), zeros(3, 2))
end

@testset "Single-precision velocities are judged against double-precision ones" begin
    exact = [3.0; 4.0; 0.0;;]
    V = Float32[3.0; 4.0; 0.5;;]
    @test _relative_velocity_error(V, exact) ≈ 0.5 / 5
end

@testset "Default forces are random but repeatable (paper §5.1)" begin
    F = _random_forces(Float64, 100)
    @test size(F) == (3, 100)
    @test F == _random_forces(Float64, 100)
    @test all(component -> -1 ≤ component ≤ 1, F)
    @test allunique(F)
    @test _random_forces(Float32, 4) isa Matrix{Float32}
end

@testset "The halving search finds the first candidate that passes" begin
    candidates = 1:100
    for boundary in (1, 2, 37, 99, 100)
        tries = Ref(0)
        found = _smallest_passing(candidates) do candidate
            tries[] += 1
            return candidate ≥ boundary
        end
        @test found == boundary
        @test tries[] ≤ ceil(Int, log2(length(candidates)))
    end
end

@testset "The halving search assumes the last candidate is good enough" begin
    tested = Float64[]
    found = _smallest_passing(0.5:0.5:1.5) do candidate
        push!(tested, candidate)
        return false
    end
    @test found == 1.5
    @test 1.5 ∉ tested
end

@testset "The halving search has nothing to search in an empty list" begin
    @test_throws ArgumentError _smallest_passing(_ -> true, Float64[])
end

# Runs the walk and also reports which candidates it timed, in order.
function _walk(times, start)
    timed = Int[]
    found = _fastest_candidate(length(times), start) do index
        push!(timed, index)
        return times[index]
    end
    return found, timed
end

@testset "The walk finds the fastest candidate of a smooth hump from any start" begin
    times = [5.0 + (index - 12)^2 for index in 1:30]
    for start in (1, 5, 12, 20, 30)
        found, timed = _walk(times, start)
        @test found == 12
        @test allunique(timed)
    end
end

@testset "The walk stops after two candidates in a row clearly slower than the best" begin
    # From the start, 1.1 times the best is not clearly slower. The two that follow are, so
    # the fast candidate beyond them is never reached.
    found, timed = _walk([10.0, 11.0, 14.0, 14.0, 1.0], 1)
    @test found == 1
    @test timed == [1, 2, 3, 4]
end

@testset "Clearly slower means 30 % slower by default, and can be changed" begin
    times = [10.0, 14.0, 14.0, 1.0]
    time_of(index) = times[index]
    @test _fastest_candidate(time_of, 4, 1) == 1
    @test _fastest_candidate(time_of, 4, 1; clearly_slower = 1.5) == 4
end

@testset "A candidate that is not clearly slower restarts the count" begin
    found, timed = _walk([10.0, 14.0, 11.0, 14.0, 1.0], 1)
    @test found == 5
    @test timed == [1, 2, 3, 4, 5]
end

@testset "The walk gets past the dips of a jagged curve (paper §5.2, Fig. 2)" begin
    # Particle timesteps per second (in millions) along the most dilute curve of Fig. 2,
    # read off the figure, from coarser grids to finer ones. Two slowdowns in a row do not
    # mean the peak has passed: the curve falls after its fourth point, then rises to its
    # true peak.
    speeds = [4.6, 5.2, 4.0, 5.3, 5.0, 5.1, 6.6, 4.5, 4.5, 3.9, 2.8, 1.8]
    found, _ = _walk(1 ./ speeds, 3)
    @test found == 7
end

@testset "The walk treats a candidate that misses the tolerance as infinitely slow" begin
    # Coarser candidates miss the tolerance and finer ones are slower, as happens in
    # practice.
    found, _ = _walk([Inf, Inf, Inf, 5.0, 4.0, 4.5, 9.0, 12.0, 15.0], 2)
    @test found == 5
end

@testset "The walk reports when no candidate it tried is usable" begin
    found, timed = _walk(fill(Inf, 9), 5)
    @test found === nothing
    @test sort(timed) == [3, 4, 5, 6, 7]
end

@testset "A single candidate needs no walking" begin
    found, timed = _walk([3.0], 1)
    @test found == 1
    @test timed == [1]
end

@testset "Taking turns picks the truly fastest while the machine's speed drifts" begin
    true_times = (a = 1.00, b = 0.97, c = 1.03)
    for drift in (0.01, -0.01)  # the machine slowly slows down, or warms up
        calls = Ref(0)
        found = _fastest_taking_turns(collect(keys(true_times))) do candidate
            calls[] += 1
            return true_times[candidate] * (1 + drift * calls[])
        end
        @test found == :b
    end
end

@testset "Taking turns never picks a candidate that misses the tolerance" begin
    found = _fastest_taking_turns(candidate -> candidate == 1 ? Inf : 2.0, [1, 2])
    @test found == 2
end

@testset "Each contender is timed twice by default, or as often as asked" begin
    calls = Ref(0)
    time_of(_) = (calls[] += 1; 1.0)
    _fastest_taking_turns(time_of, [1, 2, 3])
    @test calls[] == 6
    calls[] = 0
    _fastest_taking_turns(time_of, [1, 2, 3]; rounds = 5)
    @test calls[] == 15
end

@testset "Targets are tightened until an attempt succeeds" begin
    targets = Float64[]
    result = _tighten_until_met(1.0, :fallback) do target
        push!(targets, target)
        return length(targets) == 3 ? :settings : nothing
    end
    @test result == :settings
    @test targets ≈ [1.0, 1 / 3, 1 / 9]
end

@testset "After the last tightening the fallback is returned with a warning" begin
    targets = Float64[]
    result = @test_logs (:warn, r"reference") _tighten_until_met(1.0, :fallback) do target
        push!(targets, target)
        return nothing
    end
    @test result == :fallback
    @test targets ≈ [1.0, 1 / 3, 1 / 9, 1 / 27]
end

@testset "The number and size of the tightenings can be changed" begin
    targets = Float64[]
    result = @test_logs (:warn, r"reference") _tighten_until_met(
        1.0, :fallback; most_tightenings = 1, factor = 2.0,
    ) do target
        push!(targets, target)
        return nothing
    end
    @test result == :fallback
    @test targets ≈ [1.0, 0.5]
end

@testset "Equal grid spacing is recognised along all three axes (paper §3)" begin
    @test _has_equal_grid_spacing((10.0, 10.0, 20.0), (Int32(8), Int32(8), Int32(16)))
    @test !_has_equal_grid_spacing((10.0, 10.0, 20.0), (Int32(8), Int32(8), Int32(8)))
    @test !_has_equal_grid_spacing((10.0, 20.0, 10.0), (Int32(8), Int32(8), Int32(8)))
end

@testset "A grid size is FFT-friendly when made only of the factors 2, 3, 5 and 7" begin
    friendly = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30,
    ]
    @test filter(_is_fft_friendly, 1:30) == friendly
    @test _is_fft_friendly(2^3 * 3 * 5 * 7)
    @test !_is_fft_friendly(2^3 * 3 * 11)
    @test !_is_fft_friendly(0)
end

_factors_only_2357(M) = nextprod((2, 3, 5, 7), M) == M

@testset "Proposed grids use only the factors 2, 3, 5 and 7, from coarse to fine" begin
    grids = _fft_friendly_grids((10.0, 10.0, 10.0), Float64; available_bytes = 10^9)
    @test !isempty(grids)
    @test all(grid -> all(_factors_only_2357, grid), grids)
    @test issorted(first.(grids))
    @test allunique(grids)
    @test all(grid -> grid[1] == grid[2] == grid[3], grids)
    @test (Int32(64), Int32(64), Int32(64)) in grids
    @test (Int32(11), Int32(11), Int32(11)) ∉ grids
end

@testset "Every proposed grid is accepted in any box and either precision" begin
    for L in ((10.0, 10.0, 10.0), (10.0, 10.0, 20.0), (8.0, 12.0, 16.0), (10.0, 15.0, 25.0))
        grids = _fft_friendly_grids(L, Float64; available_bytes = 10^9)
        @test !isempty(grids)
        @test all(grid -> _has_equal_grid_spacing(L, grid), grids)
        for T in (Float32, Float64), grid in first(grids, 4)
            config = FFCMConfig{T}(;
                L = T.(L), R_c = T(minimum(L) / 4), N = 1, kernel_widths_ratio = T(1),
                viscosity = T(1), num_grid_points = grid, M_G = 2, fft_planning = :estimate,
            )
            @test config.num_grid_points == grid
        end
    end
end

@testset "A box whose sides share no usable spacing has no grid to propose" begin
    @test isempty(_fft_friendly_grids((10.0, 10.0 * sqrt(2), 10.0), Float64))
end

@testset "The memory a grid needs is what the configuration allocates" begin
    for T in (Float32, Float64)
        grid = (Int32(8), Int32(9), Int32(10))
        config = FFCMConfig{T}(;
            L = (T(8), T(9), T(10)), R_c = T(2), N = 1, kernel_widths_ratio = T(1),
            viscosity = T(1), num_grid_points = grid, M_G = 2, fft_planning = :estimate,
        )
        fields = (
            config.grid.force_density, config.grid.fluid_velocity, config.solver.fluid_hat,
        )
        @test _grid_buffer_bytes(T, grid) == sum(sum(sizeof, components(f)) for f in fields)
    end
end

@testset "Grids too large for the memory available are not proposed" begin
    L = (10.0, 10.0, 10.0)
    available_bytes = 10^8
    for T in (Float32, Float64)
        grids = _fft_friendly_grids(L, T; available_bytes)
        @test all(grid -> _grid_buffer_bytes(T, grid) ≤ available_bytes, grids)
        @test _grid_buffer_bytes(T, grids[end] .* Int32(2)) > available_bytes
    end
    @test length(_fft_friendly_grids(L, Float32; available_bytes)) >
          length(_fft_friendly_grids(L, Float64; available_bytes))
end

const _UNIT_SPHERE_σ = 1 / sqrt(π)

@testset "Accuracy ratios become configuration settings on a given grid (paper §5.2)" begin
    L = (40.0, 40.0, 40.0)
    ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
    grid = (Int32(32), Int32(32), Int32(32))
    settings = _settings_on_grid(L, 100, grid, ratios)
    Σ = 1.0 * 40 / 32
    @test settings.kernel_widths_ratio ≈ Σ / _UNIT_SPHERE_σ
    @test settings.num_grid_points == grid
    @test settings.M_G == 11
    @test settings.R_c ≈ 5.0 * Σ
end

@testset "The Gaussian is never narrower than the particle's (standard FCM, paper §2)" begin
    L = (40.0, 40.0, 40.0)
    ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
    settings = _settings_on_grid(L, 1000, (Int32(100), Int32(100), Int32(100)), ratios)
    @test settings.kernel_widths_ratio == 1
    # The correction vanishes at Σ = σ, so the cutoff only sizes the cells: about one
    # particle per cell, and never beyond half the box.
    @test settings.R_c ≈ cbrt(40.0^3 / 1000)
    single = _settings_on_grid(L, 1, (Int32(100), Int32(100), Int32(100)), ratios)
    @test single.R_c ≈ 20.0
end

@testset "Settings that do not fit the box or the grid are not proposed" begin
    L = (40.0, 40.0, 40.0)
    grid = (Int32(16), Int32(16), Int32(16))
    ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
    @test _settings_on_grid(L, 100, grid, ratios) !== nothing
    @test _settings_on_grid(L, 100, grid, (; ratios..., cutoff_ratio = 9.0)) === nothing
    @test _settings_on_grid(L, 100, grid, (; ratios..., M_G = 17)) === nothing
end

@testset "Proposed settings configure the operator in either precision" begin
    for T in (Float32, Float64)
        L = (T(40), T(40), T(40))
        ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
        grid = (Int32(32), Int32(32), Int32(32))
        settings = _settings_on_grid(L, 8, grid, ratios)
        config = FFCMConfig(;
            L, N = 8, viscosity = T(1), fft_planning = :estimate, settings...,
        )
        @test config isa FFCMConfig{T}
        @test config.Σ ≈ T(40 / 32)
    end
end

@testset "A Gaussian widened by under a quarter gives way to standard FCM (§5.2)" begin
    L = (40.0, 40.0, 40.0)
    ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
    # Σ/σ = 1.77 on the coarser grid is worth its pair work; 1.13 on the finer is not.
    coarse = _candidate_settings(L, 100, (Int32(40), Int32(40), Int32(40)), ratios)
    fine = _candidate_settings(L, 100, (Int32(63), Int32(63), Int32(63)), ratios)
    @test coarse.kernel_widths_ratio ≈ 1.0 / _UNIT_SPHERE_σ
    @test fine.kernel_widths_ratio == 1
    # With no widening counted as too little, the finer grid keeps its wider Gaussian.
    kept = _candidate_settings(
        L, 100, (Int32(63), Int32(63), Int32(63)), ratios; worthwhile_widening = 1.0,
    )
    @test kept.kernel_widths_ratio ≈ (40 / 63) / _UNIT_SPHERE_σ
end

@testset "The starting grid spacing balances grid work against pair work" begin
    # (V/N)^⅓ = 4, divided by 30^⅙ √5, worked out by hand.
    @test _balanced_grid_spacing((40.0, 40.0, 40.0), 1000, 1.0, 5.0) ≈ 1.01482 atol = 1e-4
    # It follows the distance between particles: eight times as many halves the spacing.
    @test _balanced_grid_spacing((40.0, 40.0, 40.0), 8000, 1.0, 5.0) ≈
          _balanced_grid_spacing((40.0, 40.0, 40.0), 1000, 1.0, 5.0) / 2
    # Non-cubic boxes use the volume.
    @test _balanced_grid_spacing((20.0, 40.0, 80.0), 1000, 1.0, 5.0) ≈ 1.01482 atol = 1e-4
    # Pairs 64 times dearer halve the spacing, because of the sixth root.
    @test _balanced_grid_spacing(
        (40.0, 40.0, 40.0), 1000, 1.0, 5.0; pair_to_grid_point_cost = 30 * 64,
    ) ≈ _balanced_grid_spacing((40.0, 40.0, 40.0), 1000, 1.0, 5.0) / 2
end

@testset "Accuracy is measured at a width that balances the work and fits the box" begin
    generous = _generous_ratios(1e-6)
    (; kernel_resolution, cutoff_ratio) = generous
    σ = _UNIT_SPHERE_σ
    L = (100.0, 100.0, 100.0)
    fine_enough = 0.01  # a spacing small enough not to matter

    # A dilute suspension in a roomy box: the width that balances the work.
    balanced = _balanced_grid_spacing(L, 1000, kernel_resolution, cutoff_ratio)
    @test _measuring_width(L, 1000, generous, fine_enough) ≈ kernel_resolution * balanced
    # A crowded one: at least 2σ, so that the correction is really at work.
    @test _measuring_width(L, 10^6, generous, fine_enough) ≈ 2σ
    # A small box: the generous cutoff must fit in half of it.
    small_box = (20.0, 20.0, 20.0)
    @test _measuring_width(small_box, 1, generous, fine_enough) ≈ 20 / (2 * cutoff_ratio)
    # A tiny box allows nothing wider than σ.
    tiny_box = (8.0, 8.0, 8.0)
    @test _measuring_width(tiny_box, 1, generous, fine_enough) ≈ σ
    # A machine short of memory: the grid can be no finer than the given spacing.
    @test _measuring_width(L, 10^6, generous, 2.0) ≈ kernel_resolution * 2.0
end

@testset "The generous stencil is odd and covers the requested number of widths" begin
    @test _generous_support(1.0, 10.0) == 11
    @test _generous_support(1.0, 11.0) == 11
    @test _generous_support(1.2, 11.7) == 15
    ratios = _generous_ratios(1e-6)
    @test ratios.kernel_resolution ≈ 1.222 atol = 1e-3
    @test ratios.widths_covered ≈ 11.68 atol = 1e-2
    @test ratios.cutoff_ratio ≈ 7.433 atol = 1e-3
end

@testset "Generous settings reach the accuracy they promise" begin
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    for target in (1e-4, 1e-6, 1e-8)
        ratios = _generous_ratios(target)
        s = ratios.kernel_resolution
        # Σ close to 2σ, where the correction is really at work.
        M = nextprod((2, 3, 5, 7), ceil(Int, L[1] * s / (2 * _UNIT_SPHERE_σ)))
        settings = _settings_on_grid(
            L, 150, (Int32(M), Int32(M), Int32(M)),
            (;
                kernel_resolution = s,
                M_G = _generous_support(s, ratios.widths_covered),
                ratios.cutoff_ratio,
            ),
        )
        config = FFCMConfig(;
            L, N = 150, viscosity = 1.0, fft_planning = :estimate, settings...,
        )
        V = mobility!(similar(Y), config, Y, F)
        @test _mean_relative_error(V, standard_fcm) ≤ 5 * target
    end
end

@testset "Timing a call takes the median, unmoved by an occasional slow call" begin
    calls = Ref(0)
    time_taken = _median_call_time() do
        calls[] += 1
        sleep(calls[] % 10 == 0 ? 0.05 : 0.002)
    end
    @test calls[] ≥ 4  # a lead-in, then at least three timed calls
    @test 0.001 ≤ time_taken < 0.02
end

@testset "A slow call is still timed at least three times" begin
    calls = Ref(0)
    time_taken = _median_call_time() do
        calls[] += 1
        sleep(0.3)
    end
    @test 4 ≤ calls[] ≤ 6
    @test time_taken ≥ 0.29
end

@testset "Timing makes at least one lead-in call, then as many timed calls as asked" begin
    calls = Ref(0)
    _median_call_time(; lead_in_seconds = 0.0, least_calls = 5, least_seconds = 0.0) do
        calls[] += 1
    end
    @test calls[] == 6
end

@testset "Timing spends at least the lead-in and timed seconds asked for" begin
    started = time()
    _median_call_time(
        () -> nothing; lead_in_seconds = 0.02, least_calls = 1, least_seconds = 0.05,
    )
    @test time() - started ≥ 0.07
end

@testset "The right answer agrees with independently computed standard FCM" begin
    # Σ-independence is what the two-width check relies on (paper §3), and standard FCM is
    # the thing fast FCM approximates.
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    planning = (; fft_planning = :estimate, fft_threads = 1)
    for tolerance in (1e-2, 1e-4)
        problem = CalibrationProblem(Y, F, L, tolerance; planning...)
        @test _mean_relative_error(problem.exact_velocities, standard_fcm) ≤ tolerance / 20
    end
end

@testset "The right answer is in double precision whatever is being calibrated" begin
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    problem = CalibrationProblem(
        Float32.(Y), Float32.(F), Float32.(L), 1e-3;
        fft_planning = :estimate, fft_threads = 1,
    )
    @test problem.exact_velocities isa Matrix{Float64}
    @test _mean_relative_error(problem.exact_velocities, standard_fcm) ≤ 1e-3 / 20
end

@testset "Two widths that disagree stop the calibration and say why" begin
    V = [3.0; 4.0; 0.0;;]
    largest_disagreement = 5e-6
    @test _check_reference_agreement(V, V, largest_disagreement) === nothing
    @test _check_reference_agreement(V .* (1 + 1e-6), V, largest_disagreement) === nothing
    @test_throws ArgumentError _check_reference_agreement(
        V .* 1.01, V, largest_disagreement,
    )
    # Two particles at the same position make the correction undefined.
    @test_throws ArgumentError _check_reference_agreement(
        fill(NaN, 3, 1), V, largest_disagreement,
    )
end

@testset "The reference's margins have defaults, and can be changed" begin
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    planning = (; fft_planning = :estimate, fft_threads = 1)
    tolerance = 1e-3

    problem = CalibrationProblem(Y, F, L, tolerance; planning...)
    @test problem.generous == _generous_ratios(tolerance / 100)
    @test problem.reference_uncertainty ≈ tolerance / 20

    tighter = CalibrationProblem(Y, F, L, tolerance; planning..., generous_margin = 1000)
    @test tighter.generous == _generous_ratios(tolerance / 1000)

    # Agreement closer than the generous settings can deliver is a disagreement.
    @test_throws ArgumentError CalibrationProblem(
        Y, F, L, tolerance; planning..., reference_agreement = 1e-9,
    )

    finer = CalibrationProblem(
        Y, F, L, tolerance; planning..., finer_reference_spacing = 0.5,
    )
    @test finer.reference_settings.num_grid_points[1] >
          problem.reference_settings.num_grid_points[1]
end
