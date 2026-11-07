using Test
using ForceCouplingMethod
using ForceCouplingMethod:
    CalibrationProblem,
    _candidate_settings,
    _balanced_grid_spacing,
    _fastest_accurate_settings,
    _fastest_candidate,
    _fastest_taking_turns,
    _fft_friendly_grids,
    _generous_ratios,
    _generous_support,
    _gpu_grid_bytes,
    _grid_buffer_bytes,
    _has_equal_grid_spacing,
    _index_with_spacing_at_most,
    _measure_grid,
    _measured_error,
    _measuring_width,
    _median_call_time,
    _random_forces,
    _reference_calculations,
    _relative_velocity_error,
    _settings_fit,
    _settings_on_grid,
    _smallest_accurate_ratios,
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
    @test isnan(_relative_velocity_error(ones(3, 2), zeros(3, 2)))
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

# Runs the walk and also reports which candidates it timed, in order.
function _walk(times, start)
    timed = Int[]
    found = _fastest_candidate(eachindex(times), start) do index
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
    @test _fastest_candidate(time_of, 1:4, 1) == 1
    @test _fastest_candidate(time_of, 1:4, 1; clearly_slower = 1.5) == 4
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

@testset "The walk keeps going toward finer grids until one meets the tolerance" begin
    # Later candidates are the more accurate ones, so a run of misses says nothing about
    # them.
    found, _ = _walk([Inf, Inf, Inf, Inf, 5.0, 4.0, 9.0, 12.0], 1)
    @test found == 6
end

@testset "The walk reports when no candidate it tried is usable" begin
    # Toward the finer end it goes all the way; toward the coarser end it stops after two.
    found, timed = _walk(fill(Inf, 9), 5)
    @test found === nothing
    @test sort(timed) == [3, 4, 5, 6, 7, 8, 9]
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

@testset "Proposed grids use only the factors 2, 3, 5 and 7, from coarse to fine" begin
    grids = _fft_friendly_grids((10.0, 10.0, 10.0), Float64; available_bytes = 10^9)
    friendly = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30]
    @test first.(grids)[eachindex(friendly)] == friendly
    @test issorted(first.(grids))
    @test allunique(grids)
    @test all(grid -> grid[1] == grid[2] == grid[3], grids)
    @test (Int32(2 * 3 * 5 * 7), Int32(2 * 3 * 5 * 7), Int32(2 * 3 * 5 * 7)) in grids
    @test (Int32(2 * 3 * 11), Int32(2 * 3 * 11), Int32(2 * 3 * 11)) ∉ grids
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

@testset "The room for grids on a GPU agrees with what has fitted on a card of 11 GB" begin
    fits(T, M, N) = _grid_buffer_bytes(T, (M, M, M)) ≤ _gpu_grid_bytes(T, N, 11e9)
    # The suspensions of the paper's §5.2, measured on a real card.
    @test fits(Float32, 500, 1_955_695)
    @test fits(Float64, 400, 488_923)
    @test !fits(Float64, 500, 1_955_695)
end

@testset "Particles leave less room for grids on a GPU, down to none at all" begin
    room(T, N) = _gpu_grid_bytes(T, N, 10^10)
    @test room(Float32, 10^6) < room(Float32, 1)
    @test room(Float64, 10^6) < room(Float32, 10^6)
    no_room = room(Float64, 10^9)
    @test no_room < 0
    @test isempty(
        _fft_friendly_grids((10.0, 10.0, 10.0), Float64; available_bytes = no_room),
    )
    @test _gpu_grid_bytes(Float32, 1, 100; usable_fraction = 1.0) ==
          100 - (18 * sizeof(Float32) + 8)
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

@testset "Settings that do not fit the box or the grid are recognised" begin
    L = (40.0, 40.0, 40.0)
    grid = (Int32(16), Int32(16), Int32(16))
    ratios = (; kernel_resolution = 1.0, M_G = 11, cutoff_ratio = 5.0)
    fit(ratios) = _settings_fit(_settings_on_grid(L, 100, grid, ratios), L)
    @test fit(ratios)
    @test !fit((; ratios..., cutoff_ratio = 9.0))
    @test !fit((; ratios..., M_G = 17))
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
    coarser, finer = (Int32(40), Int32(40), Int32(40)), (Int32(63), Int32(63), Int32(63))
    # Σ/σ = 1.77 on the coarser grid is worth its pair work; 1.13 on the finer is not.
    coarse = _settings_on_grid(L, 100, coarser, ratios; worthwhile_widening = 1.25)
    fine = _settings_on_grid(L, 100, finer, ratios; worthwhile_widening = 1.25)
    @test coarse.kernel_widths_ratio ≈ 1.0 / _UNIT_SPHERE_σ
    @test fine.kernel_widths_ratio == 1
    # With no widening counted as too little, the finer grid keeps its wider Gaussian.
    kept = _settings_on_grid(L, 100, finer, ratios)
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

@testset "Short of memory, the reference moves to a wider Gaussian on a coarser grid" begin
    L = (100.0, 100.0, 100.0)
    N = 10^5
    generous = _generous_ratios(1e-6)
    roomy = _fft_friendly_grids(L, Float64; available_bytes = 10^10)
    short = _fft_friendly_grids(L, Float64; available_bytes = 10^8)
    unhindered = _reference_calculations(L, N, generous, roomy, 0.8)
    squeezed = _reference_calculations(L, N, generous, short, 0.8)

    @test unhindered.coarser.num_grid_points[1] < unhindered.finer.num_grid_points[1]
    @test unhindered.finer.num_grid_points[1] < short[end][1] * 2
    @test squeezed.finer.num_grid_points == short[end]
    @test squeezed.measuring_width > unhindered.measuring_width
    @test squeezed.coarser.kernel_widths_ratio > unhindered.coarser.kernel_widths_ratio
    # The settings of the finer calculation, as `FFCMConfig` takes them in either precision.
    single = _reference_calculations(Float32.(L), N, generous, short, 0.8)
    @test single.in_callers_precision.R_c isa Float32
    @test single.in_callers_precision.num_grid_points == short[end]
    @test single.in_callers_precision.R_c ≈ squeezed.finer.R_c
end

@testset "The reference reports when the grids a device can hold are too few for it" begin
    L = (100.0, 100.0, 100.0)
    generous = _generous_ratios(1e-6)
    # A grid of 10³ at most: the Gaussian becomes so wide that its cutoff leaves the box.
    tiny = _fft_friendly_grids(L, Float64; available_bytes = 10^5)
    @test !isempty(tiny)
    @test _reference_calculations(L, 10^5, generous, tiny, 0.8) === nothing
    @test _reference_calculations(L, 10^5, generous, NTuple{3, Int32}[], 0.8) === nothing
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

@testset "The right answer stays on the processor where the GPU has no room for it" begin
    # 250 kB holds a small grid in single precision, but in double precision only a grid so
    # coarse that the Gaussian's cutoff would leave the box. No GPU is touched here, because
    # the calculations that stay on the processor are the only ones this test runs.
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    problem = CalibrationProblem(
        Float32.(Y), Float32.(F), Float32.(L), 1e-3;
        fft_planning = :estimate,
        fft_threads = 1,
        gpu_acceleration = true,
        free_gpu_bytes = 250_000,
    )
    @test problem.gpu_acceleration
    @test !problem.reference_on_gpu
    @test !isempty(problem.grids)
    @test all(grid -> _grid_buffer_bytes(Float32, grid) ≤ 0.6 * 250_000, problem.grids)
    @test _mean_relative_error(problem.exact_velocities, standard_fcm) ≤ 1e-3 / 20
end

@testset "On the processor, a calibration has nothing to do with the GPU" begin
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 20)
    F = _random_forces(Float64, 20)
    problem = CalibrationProblem(Y, F, L, 1e-2; fft_planning = :estimate, fft_threads = 1)
    @test !problem.gpu_acceleration
    @test !problem.reference_on_gpu
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

    # Agreement closer than the generous settings can deliver is a disagreement, and the
    # calibration stops and says why.
    @test_throws "differ" CalibrationProblem(
        Y, F, L, tolerance; planning..., reference_agreement = 1e-9,
    )

    finer = CalibrationProblem(
        Y, F, L, tolerance; planning..., finer_reference_spacing = 0.5,
    )
    @test finer.reference_settings.num_grid_points[1] >
          problem.reference_settings.num_grid_points[1]
end

# A problem shared by the tests of the stages below, where a search for the fastest grid has
# a real choice to make. It is small, so that the whole search takes seconds.
const _STAGE_TEST_L = (40.0, 40.0, 40.0)
const _STAGE_TEST_N = 1222
const _STAGE_TEST_TOLERANCE = 1e-3
const _STAGE_TEST_PROBLEM = CalibrationProblem(
    _non_overlapping_suspension(_STAGE_TEST_L, _STAGE_TEST_N),
    _random_forces(Float64, _STAGE_TEST_N),
    _STAGE_TEST_L,
    _STAGE_TEST_TOLERANCE;
    fft_planning = :estimate,
    fft_threads = 1,
)
const _STAGE_TEST_RATIOS = _smallest_accurate_ratios(
    _STAGE_TEST_PROBLEM, _STAGE_TEST_TOLERANCE,
)

@testset "Each search finds the smallest value meeting its share (paper §5.1, §5.2)" begin
    problem = _STAGE_TEST_PROBLEM
    target = _STAGE_TEST_TOLERANCE
    (; kernel_resolution, M_G, cutoff_ratio) = _STAGE_TEST_RATIOS
    resolution_error(s) = _measured_error(problem; kernel_resolution = s)
    @test resolution_error(kernel_resolution) ≤ target / 3
    @test resolution_error(kernel_resolution - 0.02) > target / 3

    stencil_error(m) = _measured_error(problem; kernel_resolution, M_G = m)
    allowed = resolution_error(kernel_resolution) + target / 3
    @test stencil_error(M_G) ≤ allowed
    @test stencil_error(M_G - 1) > allowed

    cutoff_error(λ) = _measured_error(problem; cutoff_ratio = λ)
    @test cutoff_error(cutoff_ratio) ≤ target / 3
    @test cutoff_error(cutoff_ratio - 0.1) > target / 3
end

@testset "The stages' values sit in the expected range for this tolerance" begin
    (; kernel_resolution, M_G, cutoff_ratio) = _STAGE_TEST_RATIOS
    @test 0.85 ≤ kernel_resolution ≤ 1.05
    @test 7 ≤ M_G ≤ 11
    @test 3.5 ≤ cutoff_ratio ≤ 5.5
end

@testset "The grids to search run from coarse to standard FCM and no further" begin
    problem = _STAGE_TEST_PROBLEM
    ratios = _STAGE_TEST_RATIOS
    candidates = _candidate_settings(problem, ratios)
    @test issorted(candidates; by = settings -> settings.num_grid_points[1])
    @test candidates[end].kernel_widths_ratio == 1
    @test candidates[1].kernel_widths_ratio > 1
    # The last grid is the first fine enough for standard FCM at this resolution.
    spacing(settings) = _STAGE_TEST_L[1] / settings.num_grid_points[1]
    σ = _UNIT_SPHERE_σ
    @test ratios.kernel_resolution * spacing(candidates[end]) ≤ σ
    @test ratios.kernel_resolution * spacing(candidates[end - 1]) > σ
    # By default a Gaussian is either standard FCM or widened by at least a quarter.
    widenings = [settings.kernel_widths_ratio for settings in candidates]
    @test all(widening -> widening == 1 || widening ≥ 1.25, widenings)
    kept = _candidate_settings(problem, ratios; worthwhile_widening = 1.0)
    @test any(settings -> 1 < settings.kernel_widths_ratio < 1.25, kept)
end

@testset "The search for the fastest grid measures every grid it times" begin
    problem = _STAGE_TEST_PROBLEM
    ratios = _STAGE_TEST_RATIOS
    outcome = _fastest_accurate_settings(problem, ratios, _STAGE_TEST_TOLERANCE)
    @test !isempty(outcome.tried)
    tried = Dict(outcome.tried)
    chosen = tried[outcome.settings]
    # The reference is trusted to its uncertainty, so a grid must beat the tolerance by
    # that much.
    largest_error = _STAGE_TEST_TOLERANCE - problem.reference_uncertainty
    @test chosen.measured_error ≤ largest_error
    @test isfinite(chosen.seconds)
    for record in values(tried)
        @test isfinite(record.seconds) == (record.measured_error ≤ largest_error)
    end
    # Near the best the times nearly tie, so an earlier, lucky time on another grid can be
    # slightly smaller than the chosen grid's.
    @test chosen.seconds ≤ 1.25 * minimum(record.seconds for record in values(tried))
end

@testset "A grid that misses the tolerance counts as infinitely slow" begin
    problem = _STAGE_TEST_PROBLEM
    ratios = _STAGE_TEST_RATIOS
    candidates = _candidate_settings(problem, ratios)
    settings = candidates[length(candidates) ÷ 2]
    measurement = _measure_grid(problem, settings, 0.0)
    @test measurement.seconds == Inf
    @test measurement.measured_error > 0
    usable = _measure_grid(problem, settings, 1.0)
    @test isfinite(usable.seconds)
    @test usable.measured_error == measurement.measured_error
end

@testset "No other grid is meaningfully faster than the chosen one" begin
    # Near the best the times nearly tie, so this allows a quarter for noise on a busy
    # machine. A badly chosen grid would still fail it.
    problem = _STAGE_TEST_PROBLEM
    ratios = _STAGE_TEST_RATIOS
    largest_error = _STAGE_TEST_TOLERANCE - problem.reference_uncertainty
    chosen = _fastest_accurate_settings(problem, ratios, _STAGE_TEST_TOLERANCE).settings
    candidates = _candidate_settings(problem, ratios)
    position = findfirst(==(chosen), candidates)
    grid_size(settings) = settings.num_grid_points[1]
    nearest_to(M) = candidates[argmin([abs(grid_size(c) - M) for c in candidates])]
    others = setdiff(
        unique([
            candidates[max(position - 1, 1)],
            candidates[min(position + 1, length(candidates))],
            nearest_to(grid_size(chosen) / 1.5),
            nearest_to(grid_size(chosen) * 1.5),
        ]),
        [chosen],
    )
    contenders = [chosen; others]
    fastest = Dict(settings => Inf for settings in contenders)
    for _ in 1:3, settings in contenders
        seconds = _measure_grid(problem, settings, largest_error).seconds
        fastest[settings] = min(fastest[settings], seconds)
    end
    @test fastest[chosen] ≤ 1.25 * minimum(fastest[other] for other in others)
end

@testset "Starting from the ansatz beats starting from the coarsest grid" begin
    # Both walks see the same times, so timing noise can't decide the comparison. Measured
    # on a few problems, starting from the ansatz spends 1.1 to 4.6 times less on the walk,
    # and the saving grows with the number of particles.
    problem = _STAGE_TEST_PROBLEM
    ratios = _STAGE_TEST_RATIOS
    largest_error = _STAGE_TEST_TOLERANCE - problem.reference_uncertainty
    # The walk's list: each grid at least a tenth larger than the one before.
    walked = _candidate_settings(problem, ratios)[1:1]
    for candidate in _candidate_settings(problem, ratios)
        if candidate.num_grid_points[1] ≥ 1.1 * walked[end].num_grid_points[1]
            push!(walked, candidate)
        end
    end
    times = Dict{Int, Float64}()
    costs = Dict{Int, Float64}()
    function walk_from(start)
        timed = Int[]
        found = _fastest_candidate(eachindex(walked), start) do index
            push!(timed, index)
            haskey(times, index) && return times[index]
            costs[index] = @elapsed begin
                times[index] = _measure_grid(problem, walked[index], largest_error).seconds
            end
            return times[index]
        end
        return found, timed
    end
    balanced_spacing = _balanced_grid_spacing(
        _STAGE_TEST_L, _STAGE_TEST_N, ratios.kernel_resolution, ratios.cutoff_ratio,
    )
    grids = [candidate.num_grid_points for candidate in walked]
    start = _index_with_spacing_at_most(_STAGE_TEST_L, grids, balanced_spacing)
    @test start > 1
    found_ansatz, timed_ansatz = walk_from(start)
    found_coarsest, timed_coarsest = walk_from(1)
    @test length(timed_ansatz) < length(timed_coarsest)
    @test sum(costs[i] for i in timed_ansatz) < sum(costs[i] for i in timed_coarsest)
    @test times[found_ansatz] ≤ 1.25 * times[found_coarsest]
end

# Calibrates without printing the report, and checks that a report was logged.
_quietly_calibrated(Y; keywords...) =
    @test_logs (:info, r"Σ/Δx") match_mode = :any ForceCouplingMethod.calibrate(
        Y; keywords...,
    )

function _calibration_test_problems()
    random_box = (24.0, 24.0, 24.0)
    non_cubic_box = (16.0, 16.0, 32.0)
    return (
        (; L = random_box, Y = _non_overlapping_suspension(random_box, 150)),
        (; L = (16.0, 16.0, 16.0), Y = [8.0; 8.0; 8.0;;]),
        (; L = non_cubic_box, Y = _non_overlapping_suspension(non_cubic_box, 390)),
    )
end

@testset "Calibrated settings meet the tolerance without overdoing it" begin
    # The check is standard FCM computed independently, not the calibration's own reference.
    for (; L, Y) in _calibration_test_problems(), tolerance in (1e-2, 1e-3, 1e-4)
        F = _random_forces(Float64, size(Y, 2))
        standard_fcm = _standard_fcm_velocities(Y, F, L)
        settings = _quietly_calibrated(
            Y; L, relative_tolerance = tolerance, F, fft_planning = :estimate,
        )
        config = FFCMConfig(;
            L, N = size(Y, 2), viscosity = 1.0, fft_planning = :estimate, settings...,
        )
        velocity_error = _mean_relative_error(
            mobility!(similar(Y), config, Y, F), standard_fcm,
        )
        @test velocity_error ≤ tolerance
        @test velocity_error ≥ tolerance / 30
    end
end

@testset "Calibrating in single precision meets the tolerance" begin
    L = (24.0, 24.0, 24.0)
    Y = _non_overlapping_suspension(L, 150)
    F = _random_forces(Float64, 150)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    L_single, Y_single, F_single = Float32.(L), Float32.(Y), Float32.(F)
    settings = _quietly_calibrated(
        Y_single;
        L = L_single, relative_tolerance = 1e-3, F = F_single, fft_planning = :estimate,
    )
    config = FFCMConfig(;
        L = L_single, N = 150, viscosity = 1.0f0, fft_planning = :estimate, settings...,
    )
    V = mobility!(similar(Y_single), config, Y_single, F_single)
    @test _mean_relative_error(V, standard_fcm) ≤ 1e-3
end

@testset "A box barely large enough for the Gaussian still ends within tolerance" begin
    # On a problem this small the grid sizes are far apart, and the first targets may miss.
    L = (8.0, 8.0, 8.0)
    Y = [2.0 6.0; 4.0 4.0; 4.0 4.0]
    F = _random_forces(Float64, 2)
    standard_fcm = _standard_fcm_velocities(Y, F, L)
    settings = _quietly_calibrated(
        Y; L, relative_tolerance = 1e-2, F, fft_planning = :estimate,
    )
    config = FFCMConfig(; L, N = 2, viscosity = 1.0, fft_planning = :estimate, settings...)
    V = mobility!(similar(Y), config, Y, F)
    @test _mean_relative_error(V, standard_fcm) ≤ 1e-2
end
