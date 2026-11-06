"""
    calibrate(
        Y; L, relative_tolerance = 1e-4, F = random forces, fft_planning = :measure,
        fft_threads = 1, gpu_acceleration = false,
    ) -> NamedTuple

Choose the numerical settings of the Fast FCM mobility operator for these particle positions
on this machine: the fastest settings found whose velocities stay within
`relative_tolerance` of standard FCM. The result goes straight into `FFCMConfig`:

    settings = ForceCouplingMethod.calibrate(Y; L, relative_tolerance = 1e-4)
    config = FFCMConfig(; L, N = size(Y, 2), viscosity = 1.0, settings...)

This repeats the procedure of Su & Keaveny (2024), §5, on the caller's own problem. The
error is the paper's (§5.1, equation (69)): `|ΔV|/|V|` for each particle, averaged over the
particles, measured against a very accurate calculation done once in double precision. The
grid resolution `Σ/Δx` and the stencil width `M_G` are found first (§5.1), then the reach of
the pair correction `R_c/Σ` (§5.2), each allowed a third of the tolerance. The grid is then
chosen by timing real calls (§5.2): every grid tried has its error measured, so settings
that miss the tolerance are never returned. Where widening the Gaussian would save too
little to pay for the pair work, standard FCM is used instead (`Σ = σ`, §5.2).

The settings suit the positions given, so calibrate again when density, clustering or `N`
change a lot. Without `F`, the error refers to random forces. A report in the paper's
notation is logged with `@info`, which Julia's standard logging tools can silence.

# Arguments
- `Y::AbstractMatrix{T}`: the `3xN` particle positions, with `T` either `Float32` or
  `Float64`. Read-only.

# Keywords
- `L::NTuple{3, T}`: the domain lengths `(L_x, L_y, L_z)`, as for `FFCMConfig`. The smallest
  must be large enough to hold the Gaussian, about 6.6 sphere radii at the default
  tolerance.
- `relative_tolerance::Real = 1e-4`: the largest acceptable error. At most `1e-2`, and at
  least `5e-5` in `Float32` or `1e-10` in `Float64`, below which rounding hides the error.
- `F::AbstractMatrix{T}`: the `3xN` forces to measure the error with. Read-only. Not all
  zero. By default random and repeatable, as in the paper's error measurements (§5.1).
- `fft_planning::Symbol = :measure`, `fft_threads::Integer = 1`: as for `FFCMConfig`. They
  change which grid is fastest, so they should match what will be run.
- `gpu_acceleration::Bool = false`: as for `FFCMConfig`. The GPU is not yet implemented
  here.

# Returns
- `NamedTuple` with `kernel_widths_ratio::T`, `num_grid_points::NTuple{3, Int32}`,
  `M_G::Int`, and `R_c::T`.

# Throws
- `ArgumentError`: if `T` is neither `Float32` nor `Float64`; `Y` is not `3xN` with `N ≥ 1`;
  `F` has a different size or is all zero; `Y` or `F` holds a non-finite value; `L` is not
  positive; `relative_tolerance` is outside its range; an FFT option is invalid; the box is
  too small to hold the Gaussian; no equal-spacing grid fits in memory; or the very accurate
  calculation at two Gaussian widths disagrees, as when two particles share a position.
- `ErrorException`: if `gpu_acceleration = true`, which is not yet implemented.
"""
function calibrate(
    Y::AbstractMatrix{T};
    L::NTuple{3, T},
    relative_tolerance::Real = 1e-4,
    F::AbstractMatrix{T} = _random_forces(T, size(Y, 2)),
    fft_planning::Symbol = :measure,
    fft_threads::Integer = 1,
    gpu_acceleration::Bool = false,
) where {T <: AbstractFloat}
    tolerance = Float64(relative_tolerance)
    _validate_calibration_request(Y, F, L, tolerance, fft_planning, fft_threads)
    gpu_acceleration && error("calibrate on the GPU is not yet implemented")
    problem = CalibrationProblem(Y, F, L, tolerance; fft_planning, fft_threads)
    @info "calibrate: computed the very accurate reference velocities"
    fallback = (; settings = problem.reference_settings, tried = [])
    outcome = _tighten_until_met(tolerance, fallback) do target
        ratios = _smallest_accurate_ratios(problem, target)
        @info "calibrate: found M_G = $(ratios.M_G), " *
              "Σ/Δx = $(round(ratios.kernel_resolution; sigdigits = 3)) " *
              "and λ_ε = $(round(ratios.cutoff_ratio; sigdigits = 3)); now timing grids"
        return _fastest_accurate_settings(problem, ratios, tolerance)
    end
    @info _calibration_report(problem, outcome, tolerance)
    return outcome.settings
end

"""
    _validate_calibration_request(Y, F, L, tolerance, fft_planning, fft_threads)

Enforces the `calibrate` preconditions before any calculation starts, throwing
`ArgumentError` on the first violation, so that a bad request fails at once and not minutes
later. The smallest tolerances are where rounding starts to hide the error, which is about
`3e-6` per particle in `Float32` and far below `1e-10` in `Float64`. The largest is the
loosest the paper covers (§5.1). Whether the box can hold the Gaussian is checked by
`CalibrationProblem`, which works out how much of it has to fit.

# Arguments
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the positions and forces.
- `L::NTuple{3, T}`: the domain lengths.
- `tolerance::Float64`: the requested relative tolerance.
- `fft_planning::Symbol`, `fft_threads::Integer`: the FFT choices.
"""
function _validate_calibration_request(
    Y::AbstractMatrix{T},
    F::AbstractMatrix{T},
    L::NTuple{3, T},
    tolerance::Float64,
    fft_planning::Symbol,
    fft_threads::Integer,
) where {T}
    T in (Float32, Float64) || throw(ArgumentError(
        "calibrate works in Float32 or Float64, like the FFTs it times; got $(T)",
    ))
    (size(Y, 1) == 3 && size(Y, 2) ≥ 1) || throw(ArgumentError(
        "Y must be 3xN with at least one particle; got size $(size(Y))",
    ))
    size(F) == size(Y) || throw(ArgumentError(
        "F must have the same size as Y; got size(F) = $(size(F)), size(Y) = $(size(Y))",
    ))
    (all(isfinite, Y) && all(isfinite, F)) || throw(ArgumentError(
        "Y and F must hold only finite values",
    ))
    any(!iszero, F) || throw(ArgumentError(
        "F must not be all zero: with no particle moving, the relative error is undefined",
    ))
    all(>(zero(T)), L) || throw(ArgumentError(
        "L components must be positive; got L = $(L)",
    ))
    smallest_tolerance = T === Float32 ? 5e-5 : 1e-10
    smallest_tolerance ≤ tolerance ≤ 1e-2 || throw(ArgumentError(
        "relative_tolerance must be between $(smallest_tolerance) and 0.01 in $(T); " *
        "got $(tolerance)",
    ))
    _validate_fft_options(fft_planning, fft_threads)
    return nothing
end

"""
The suspension being calibrated, with the velocities every trial is compared with.

# Fields
- `L::NTuple{3, T}`: the domain lengths.
- `Y::Matrix{T}`, `F::Matrix{T}`: the `3xN` positions and forces.
- `fft_planning::Symbol`, `fft_threads::Int`: the FFT choices timed grids are built with.
- `grids::Vector{NTuple{3, Int32}}`: the grids worth trying in precision `T`, from coarse to
  fine.
- `generous::@NamedTuple`: the values of `_generous_ratios` for this tolerance.
- `measuring_width::Float64`: the Gaussian width `Σ` accuracy trials are measured at.
- `exact_velocities::Matrix{Float64}`: the `3xN` velocities taken as exact.
- `reference_uncertainty::Float64`: how far they may be from the exact answer, which is the
  disagreement allowed between the two Gaussian widths. Settings must beat the tolerance by
  this much to be sure of meeting it.
- `reference_settings::@NamedTuple`: the settings that produced them, as `FFCMConfig`
  keywords in precision `T`.
"""
struct CalibrationProblem{T <: AbstractFloat}
    L::NTuple{3, T}
    Y::Matrix{T}
    F::Matrix{T}
    fft_planning::Symbol
    fft_threads::Int
    grids::Vector{NTuple{3, Int32}}
    generous::@NamedTuple{
        kernel_resolution::Float64, widths_covered::Float64, cutoff_ratio::Float64,
    }
    measuring_width::Float64
    exact_velocities::Matrix{Float64}
    reference_uncertainty::Float64
    reference_settings::@NamedTuple{
        kernel_widths_ratio::T, num_grid_points::NTuple{3, Int32}, M_G::Int, R_c::T,
    }
end

"""
    CalibrationProblem(
        Y, F, L, tolerance; fft_planning, fft_threads, generous_margin = 100,
        reference_agreement = 1 / 20, finer_reference_spacing = 0.8,
    ) -> CalibrationProblem

Gather the suspension and compute the velocities every trial is compared with: FFCM with
generous settings, in double precision whatever is being calibrated, so that the reference's
own rounding never counts against the settings being judged. This stands in for the very
accurate calculation of Su & Keaveny (2024), §5.1. It is computed at two Gaussian widths,
because the exact answer doesn't depend on `Σ/σ` (§3). In a box too small to allow anything
wider than `σ`, it is simply standard FCM.

# Arguments
- `Y::AbstractMatrix{T}`, `F::AbstractMatrix{T}`: the `3xN` positions and forces.
- `L::NTuple{3, T}`: the domain lengths.
- `tolerance::Float64`: the error the settings being calibrated are allowed.

# Keywords
- `fft_planning::Symbol`, `fft_threads::Integer`: the FFT choices timed grids are built
  with. The reference plans instantly, since its velocities don't depend on the planner.
- `generous_margin::Real = 100`: how many times tighter than `tolerance` the generous
  settings aim. The reference then stays about a hundredth of the tolerance from the exact
  answer, which is small beside the third of the tolerance each search works with, so it
  hardly disturbs what they measure. Aiming tighter would cost more per trial and gain
  nothing.
- `reference_agreement::Real = 1 / 20`: the fraction of `tolerance` the two widths may
  differ by. They usually differ by about a hundredth, so this leaves room for unusual
  suspensions while keeping nearly all of the tolerance for the settings being judged.
- `finer_reference_spacing::Real = 0.8`: the grid spacing of the second calculation relative
  to the first. Its Gaussian is 20 % narrower, which is different enough to expose any
  dependence on the width, and it needs about twice the grid points, which keeps it cheap.

# Returns
- `CalibrationProblem{T}`.

# Throws
- `ArgumentError`: if the box is too small to hold the Gaussian, if no equal-spacing grid
  fits in memory, if the generous settings don't fit the box and the memory, or if the two
  widths disagree or are not finite, as when two particles share a position and the
  correction between them is undefined.
"""
function CalibrationProblem(
    Y::AbstractMatrix{T},
    F::AbstractMatrix{T},
    L::NTuple{3, T},
    tolerance::Float64;
    fft_planning::Symbol,
    fft_threads::Integer,
    generous_margin::Real = 100,
    reference_agreement::Real = 1 / 20,
    finer_reference_spacing::Real = 0.8,
) where {T}
    N = size(Y, 2)
    L_double = Float64.(L)
    grids_double = _fft_friendly_grids(L_double, Float64)
    isempty(grids_double) && throw(ArgumentError(
        "no grid fits in memory with equal spacing along all axes and sizes made of the " *
        "factors 2, 3, 5 and 7, in a box of size $(L); a simpler ratio between the " *
        "sides of the box would allow one",
    ))
    generous = _generous_ratios(tolerance / generous_margin)
    # A Gaussian cut off by the box would give the same wrong answer at both widths, so the
    # check of the reference below can't catch this.
    smallest_box_side = generous.widths_covered * _unit_sphere_kernel_width(Float64)
    minimum(L_double) ≥ smallest_box_side || throw(ArgumentError(
        "the box is too small to hold the Gaussian: at this tolerance its smallest side " *
        "must be at least $(smallest_box_side) sphere radii; got L = $(L)",
    ))
    # The second reference grid is finer still, so the first has to leave it room in memory.
    finest_spacing = L_double[1] / grids_double[end][1]
    measuring_width = _measuring_width(
        L_double, N, generous, finest_spacing / finer_reference_spacing,
    )
    ratios = (;
        generous.kernel_resolution,
        M_G = _generous_support(generous.kernel_resolution, generous.widths_covered),
        generous.cutoff_ratio,
    )
    spacing = measuring_width / generous.kernel_resolution
    coarser_index = _index_with_spacing_at_most(L_double, grids_double, spacing)
    finer_index = _index_with_spacing_at_most(
        L_double, grids_double, finer_reference_spacing * spacing,
    )
    coarser = _settings_on_grid(L_double, N, grids_double[coarser_index], ratios)
    finer = _settings_on_grid(L_double, N, grids_double[finer_index], ratios)
    reference_settings = _settings_on_grid(L, N, grids_double[finer_index], ratios)
    fit = _settings_fit(coarser, L_double) && _settings_fit(finer, L_double) &&
          _settings_fit(reference_settings, L)
    fit || throw(ArgumentError(
        "there is no room for the very accurate reference calculation: its grid or " *
        "its cutoff doesn't fit the available memory and a box of size $(L)",
    ))
    Y_double, F_double = Matrix{Float64}(Y), Matrix{Float64}(F)
    planning = (; fft_planning = :estimate, fft_threads)
    V_coarser = _mobility_call(L_double, Y_double, F_double, coarser; planning...)()
    V_finer = _mobility_call(L_double, Y_double, F_double, finer; planning...)()
    reference_uncertainty = Float64(reference_agreement * tolerance)
    disagreement = _relative_velocity_error(V_coarser, V_finer)
    disagreement ≤ reference_uncertainty || throw(ArgumentError(
        "the very accurate velocities computed at two Gaussian widths differ by " *
        "$(disagreement), more than the $(reference_uncertainty) allowed, although the " *
        "exact answer doesn't depend on the width. Check that no two particles share a " *
        "position",
    ))
    return CalibrationProblem{T}(
        L,
        Matrix{T}(Y),
        Matrix{T}(F),
        fft_planning,
        Int(fft_threads),
        _fft_friendly_grids(L_double, T),
        generous,
        measuring_width,
        V_finer,
        reference_uncertainty,
        reference_settings,
    )
end

"""
    _measured_error(problem; kernel_resolution, M_G, cutoff_ratio) -> Float64

One accuracy trial: the error against the reference, at the problem's measuring width. Each
setting not given is generous, so that a trial studies one source of error while the others
stay negligible. The grid is the coarsest that keeps `Σ` within that width, and the FFT is
planned instantly, since the velocities don't depend on the planner.

# Arguments
- `problem::CalibrationProblem`: the suspension and its reference.

# Keywords
- `kernel_resolution::Real`: the grid resolution `Σ/Δx`.
- `M_G::Integer`: the stencil width. By default it covers a generous number of Gaussian
  widths at `kernel_resolution`, because it has to grow with `Σ/Δx`: at a fixed `M_G`, the
  error turns back up as `Σ/Δx` grows.
- `cutoff_ratio::Real`: the reach of the pair correction `λ_ε = R_c/Σ`.

# Returns
- `Float64`: the error, or `Inf` if these settings don't fit the box or the grid.
"""
function _measured_error(
    problem::CalibrationProblem;
    kernel_resolution::Real = problem.generous.kernel_resolution,
    M_G::Integer = _generous_support(kernel_resolution, problem.generous.widths_covered),
    cutoff_ratio::Real = problem.generous.cutoff_ratio,
)
    (; L, Y, F, grids, fft_threads) = problem
    spacing = problem.measuring_width / kernel_resolution
    grid = grids[_index_with_spacing_at_most(L, grids, spacing)]
    ratios = (; kernel_resolution, M_G, cutoff_ratio)
    settings = _settings_on_grid(L, size(Y, 2), grid, ratios)
    _settings_fit(settings, L) || return Inf
    V = _mobility_call(L, Y, F, settings; fft_planning = :estimate, fft_threads)()
    return _relative_velocity_error(V, problem.exact_velocities)
end

"""
    _smallest_accurate_ratios(
        problem, target; smallest_resolution = 0.5, resolution_step = 0.02,
        smallest_cutoff_ratio = 1.0, cutoff_ratio_step = 0.1,
    ) -> NamedTuple

The first two stages of the procedure of Su & Keaveny (2024), §5. In the first (§5.1, Fig.
1(a)), the smallest `Σ/Δx` whose error stays within a third of `target`, with the pair
cutoff generous and the stencil covering a generous number of Gaussian widths, then the
smallest `M_G` at that `Σ/Δx` that adds at most another third. In the second (§5.2, Fig.
1(b)), the smallest `λ_ε = R_c/Σ` whose error stays within a third of `target`, with the
grid part generous. There are three separate sources of error, so three thirds keep the
total within `target` even in the worst case.

# Arguments
- `problem::CalibrationProblem`: the suspension and its reference.
- `target::Float64`: the error the three searches share.

# Keywords
- `smallest_resolution::Real = 0.5`: where the search for `Σ/Δx` begins. This is well below
  what even the loosest tolerance needs, since the error there is already several percent.
- `resolution_step::Real = 0.02`: the step of that search. A step changes the number of grid
  points by about 6 %, so a finer one would hardly change the cost.
- `smallest_cutoff_ratio::Real = 1.0`: where the search for `λ_ε` begins. A very few
  particles in a large box can need no more than this.
- `cutoff_ratio_step::Real = 0.1`: the step of that search. Near typical values a step
  changes the number of pairs by about 6 %.

# Returns
- `NamedTuple` with `kernel_resolution` (`Σ/Δx`), `M_G`, and `cutoff_ratio` (`λ_ε`).
"""
function _smallest_accurate_ratios(
    problem::CalibrationProblem,
    target::Float64;
    smallest_resolution::Real = 0.5,
    resolution_step::Real = 0.02,
    smallest_cutoff_ratio::Real = 1.0,
    cutoff_ratio_step::Real = 0.1,
)
    (; generous) = problem
    share = target / 3

    resolutions = smallest_resolution:resolution_step:generous.kernel_resolution
    kernel_resolution = _smallest_passing(resolutions) do resolution
        return _measured_error(problem; kernel_resolution = resolution) ≤ share
    end

    # The stencil is searched at the chosen resolution, because what cuts off the Gaussian
    # is how many widths it covers there.
    allowed = _measured_error(problem; kernel_resolution) + share
    stencils = 2:_generous_support(kernel_resolution, generous.widths_covered)
    M_G = _smallest_passing(stencils) do stencil
        return _measured_error(problem; kernel_resolution, M_G = stencil) ≤ allowed
    end

    cutoff_ratios = smallest_cutoff_ratio:cutoff_ratio_step:generous.cutoff_ratio
    cutoff_ratio = _smallest_passing(cutoff_ratios) do candidate
        return _measured_error(problem; cutoff_ratio = candidate) ≤ share
    end

    return (; kernel_resolution, M_G, cutoff_ratio)
end

"""
    _candidate_settings(problem, ratios; worthwhile_widening = 1.25) -> Vector{<:NamedTuple}

The settings the search for the fastest grid may use, one per grid, from coarse to fine:
those that fit the box and the grid, ending at the first grid fine enough for standard FCM
at this resolution, since anything finer would only cost more (paper §5.2).

# Arguments
- `problem::CalibrationProblem`: the suspension and its grids.
- `ratios::NamedTuple`: as for `_settings_on_grid`.

# Keywords
- `worthwhile_widening::Real = 1.25`: the `Σ/σ` below which standard FCM takes over. So
  little widening saves little grid work, yet it needs pair work that standard FCM avoids.
  The paper makes this point for dense suspensions (§5.2). At about this widening, standard
  FCM on the same grid usually still meets the tolerance, because we measured its grid error
  to be 5 to 40 times smaller at the same spacing. Where it doesn't, the search for the
  fastest grid finds that out, since it measures every grid's error.

# Returns
- `Vector` of settings, as from `_settings_on_grid`. Possibly empty.
"""
function _candidate_settings(
    problem::CalibrationProblem, ratios::NamedTuple; worthwhile_widening::Real = 1.25,
)
    (; L, Y, grids) = problem
    σ = _unit_sphere_kernel_width(Float64)
    last_useful = _index_with_spacing_at_most(L, grids, σ / ratios.kernel_resolution)
    candidates = map(grids[1:last_useful]) do grid
        return _settings_on_grid(L, size(Y, 2), grid, ratios; worthwhile_widening)
    end
    return filter(settings -> _settings_fit(settings, L), candidates)
end

"""
    _measure_grid(problem, settings, largest_error) -> NamedTuple

Build a configuration with these settings and the caller's FFT choices, since the FFT plans
decide how fast the grid is, measure its error against the reference, and time it if the
error is acceptable. A grid that misses counts as infinitely slow, which keeps failing
settings out of the search for the fastest without any special case there.

# Arguments
- `problem::CalibrationProblem`: the suspension and its reference.
- `settings::NamedTuple`: `FFCMConfig` keywords, as from `_candidate_settings`.
- `largest_error::Float64`: the largest acceptable error.

# Returns
- `NamedTuple` with `measured_error::Float64` and `seconds::Float64`, the median time of one
  call, or `Inf` if the error is too large.
"""
function _measure_grid(
    problem::CalibrationProblem, settings::NamedTuple, largest_error::Float64,
)
    (; L, Y, F, fft_planning, fft_threads) = problem
    call = _mobility_call(L, Y, F, settings; fft_planning, fft_threads)
    measured_error = _relative_velocity_error(call(), problem.exact_velocities)
    measured_error ≤ largest_error || return (; measured_error, seconds = Inf)
    return (; measured_error, seconds = _median_call_time(call))
end

"""
    _fastest_accurate_settings(
        problem, ratios, tolerance; warm_up_seconds = 1.0, grid_step = 0.1,
    ) -> Union{NamedTuple, Nothing}

The third stage of the procedure of Su & Keaveny (2024), §5: with the accuracy ratios known,
the grid fixes everything else, so the fastest grid is found by timing real calls (§5.2,
Fig. 2, Table 3). The search begins at the ansatz of `_balanced_grid_spacing`, walks to
either side, and then compares every grid size close to the best, taking turns, since FFT
speed varies between sizes. Every grid has its error measured, and must beat the tolerance
by the reference's uncertainty, which guarantees the true error is within the tolerance.

# Arguments
- `problem::CalibrationProblem`: the suspension and its reference.
- `ratios::NamedTuple`: as for `_settings_on_grid`.
- `tolerance::Float64`: the largest acceptable error.

# Keywords
- `warm_up_seconds::Real = 1.0`: how long to run calls, untimed, before timing anything. A
  cold machine is slow for its first moments. A second is enough for a processor to reach
  full speed.
- `grid_step::Real = 0.1`: how much larger each grid walked is than the one before, as a
  fraction of its size. The time changes little over a tenth of the grid size near the best,
  so such steps find the region of the fastest grid with few grids timed. It is also how
  close to the best a grid must be to enter the final comparison, which therefore covers the
  sizes the walk skipped.

# Returns
- `NamedTuple` with `settings`, the `FFCMConfig` keywords of the fastest grid, and `tried`,
  a vector of `settings => (; measured_error, seconds)` pairs from coarse to fine, with each
  grid's fastest time. Or `nothing` if no grid tried met the tolerance.
"""
function _fastest_accurate_settings(
    problem::CalibrationProblem, ratios::NamedTuple, tolerance::Float64;
    warm_up_seconds::Real = 1.0, grid_step::Real = 0.1,
)
    (; L, Y, F, fft_planning, fft_threads) = problem
    candidates = _candidate_settings(problem, ratios)
    isempty(candidates) && return nothing
    grid_size(settings) = settings.num_grid_points[1]
    largest_error = tolerance - problem.reference_uncertainty
    tried = Dict{
        eltype(candidates), @NamedTuple{measured_error::Float64, seconds::Float64},
    }()
    function time_on(settings)
        (; measured_error, seconds) = _measure_grid(problem, settings, largest_error)
        earlier_seconds = haskey(tried, settings) ? tried[settings].seconds : Inf
        tried[settings] = (; measured_error, seconds = min(earlier_seconds, seconds))
        return seconds
    end

    walked = candidates[1:1]
    for candidate in candidates
        if grid_size(candidate) ≥ (1 + grid_step) * grid_size(walked[end])
            push!(walked, candidate)
        end
    end
    balanced_spacing = _balanced_grid_spacing(
        L, size(Y, 2), ratios.kernel_resolution, ratios.cutoff_ratio,
    )
    walked_grids = [candidate.num_grid_points for candidate in walked]
    start = _index_with_spacing_at_most(L, walked_grids, balanced_spacing)
    _median_call_time(
        _mobility_call(L, Y, F, walked[start]; fft_planning, fft_threads);
        lead_in_seconds = warm_up_seconds, least_calls = 1, least_seconds = 0,
    )
    best = _fastest_candidate(time_on, walked, start)
    best === nothing && return nothing

    best_size = grid_size(best)
    nearby = filter(candidates) do candidate
        return abs(grid_size(candidate) - best_size) ≤ grid_step * best_size
    end
    fastest = _fastest_taking_turns(time_on, nearby)
    return (;
        settings = fastest,
        tried = sort!(collect(tried); by = pair -> grid_size(pair.first)),
    )
end

"""
    _calibration_report(problem, outcome, tolerance) -> String

The text of the report `calibrate` logs: the settings in the paper's notation (§5), the
measured error, the time per call, the particle timesteps per second (§5.2), and every grid
tried.

# Arguments
- `problem::CalibrationProblem`: the suspension.
- `outcome::NamedTuple`: `settings` and `tried`, as from `_fastest_accurate_settings`. An
  empty `tried` means the settings are the reference's, returned because nothing met the
  tolerance.
- `tolerance::Float64`: the requested relative tolerance.

# Returns
- `String`: the report.
"""
function _calibration_report(
    problem::CalibrationProblem, outcome::NamedTuple, tolerance::Float64,
)
    (; L, Y) = problem
    (; settings, tried) = outcome
    N = size(Y, 2)
    σ = _unit_sphere_kernel_width(Float64)
    rounded(x) = round(Float64(x); sigdigits = 3)
    grid = settings.num_grid_points
    Σ_over_σ = settings.kernel_widths_ratio
    Σ = Σ_over_σ * σ
    R_c = settings.R_c

    lines = ["calibrate: $(N) particles, relative tolerance ε = $(tolerance)"]
    isempty(tried) && push!(
        lines,
        "No settings searched met the tolerance, so these are the very accurate " *
        "reference settings, which are correct but slow:",
    )
    standard_fcm = Σ_over_σ == 1
    cutoff = standard_fcm ? "" : ", λ_ε = R_c/Σ = $(rounded(R_c / Σ))"
    widening = standard_fcm ? "standard FCM (Σ = σ), which needs no pair correction" :
               "Σ/σ = $(rounded(Σ_over_σ)), R_c/σ = $(rounded(R_c / σ))"
    push!(
        lines,
        "Settings (paper §5): M_G = $(settings.M_G), " *
        "Σ/Δx = $(rounded(Σ * grid[1] / L[1]))$(cutoff)",
        "Grid $(join(grid, "×")): $(widening)",
    )
    isempty(tried) && return join(lines, "\n")

    chosen = Dict(tried)[settings]
    push!(
        lines,
        "Measured error $(round(chosen.measured_error; sigdigits = 2)), " *
        "$(rounded(1e3 * chosen.seconds)) ms per call, " *
        "$(round(Int, N / chosen.seconds)) particle timesteps per second (paper §5.2)",
        "Grids tried, from coarse to fine:",
    )
    for (candidate, record) in tried
        speed = isfinite(record.seconds) ? "$(rounded(1e3 * record.seconds)) ms" :
                "missed the tolerance"
        push!(
            lines,
            "  $(join(candidate.num_grid_points, "×")): " *
            "Σ/σ = $(rounded(candidate.kernel_widths_ratio)), " *
            "error $(round(record.measured_error; sigdigits = 2)), $(speed)",
        )
    end
    return join(lines, "\n")
end

"""
    _mobility_call(L, Y, F, settings; fft_planning, fft_threads) -> Function

A function of no arguments that applies the mobility operator with these settings. It holds
the configuration, so dropping it frees the configuration, which matters when many large
ones are tried in a row.

# Arguments
- `L::NTuple{3, T}`: the domain lengths.
- `Y::Matrix{T}`, `F::Matrix{T}`: the `3xN` positions and forces.
- `settings::NamedTuple`: `FFCMConfig` keywords, as from `_settings_on_grid`.

# Keywords
- `fft_planning::Symbol`, `fft_threads::Integer`: as for `FFCMConfig`.

# Returns
- A function returning the `3xN` velocities as a `Matrix{T}`, at unit viscosity, since
  relative errors and times don't depend on the viscosity. Each call overwrites that matrix.
"""
function _mobility_call(
    L::NTuple{3, T}, Y::Matrix{T}, F::Matrix{T}, settings::NamedTuple;
    fft_planning::Symbol, fft_threads::Integer,
) where {T}
    config = FFCMConfig{T}(;
        L, N = size(Y, 2), viscosity = one(T), fft_planning, fft_threads, settings...,
    )
    V = similar(Y)
    # The backend is chosen at run time, so the compiler can't know the configuration's type
    # here, but the closure captures its exact type, so what it runs is fully specialised.
    return () -> mobility!(V, config, Y, F)
end

"""
    _relative_velocity_error(V, exact) -> Float64

The error measure of Su & Keaveny (2024), §5.1 equation (69): `|V_n - exact_n| / |exact_n|`
for each particle `n`, averaged over the particles. A particle whose exact velocity is zero
has no relative error and is left out of the average. Exact here just means computed with a
very accurate method, it is not actually exact.

# Arguments
- `V::AbstractMatrix`: the `3xN` velocities being judged, in any precision.
- `exact::AbstractMatrix`: the `3xN` velocities taken as exact.

# Returns
- `Float64`: the mean relative error, or `NaN` if no particle moves.
"""
function _relative_velocity_error(V::AbstractMatrix, exact::AbstractMatrix)
    total = 0.0
    moving_particles = 0
    for n in axes(exact, 2)
        speed² = sum(abs2(Float64(exact[i, n])) for i in axes(exact, 1))
        iszero(speed²) && continue
        difference² = sum(
            abs2(Float64(V[i, n]) - Float64(exact[i, n])) for i in axes(exact, 1)
        )
        total += sqrt(difference² / speed²)
        moving_particles += 1
    end
    return total / moving_particles
end

"""
    _random_forces(T, N) -> Matrix{T}

The default forces of a calibration: random, as in the paper's error measurements (§5.1),
and the same on every call, so that a calibration can be repeated. Forces aren't known in
advance inside a resistance solve, and random ones exercise every direction.

# Arguments
- `T::Type{<:AbstractFloat}`: the precision.
- `N::Integer`: the number of particles.

# Returns
- `Matrix{T}` of size `3xN`, each component uniform in `[-1, 1]`.
"""
function _random_forces(::Type{T}, N::Integer) where {T <: AbstractFloat}
    return T.(2 .* rand(Xoshiro(2024), 3, N) .- 1)
end

"""
    _smallest_passing(passes, candidates) -> eltype(candidates)

The first entry of `candidates` for which `passes` is true, found by halving the range at
each try. It relies on `passes` turning from false to true once along the list, which holds
here because the error falls as a setting grows (paper §5.1, §5.2). The last candidate is
never tried: it is returned if nothing before it passes, so it should be one already known
to be good enough.

# Arguments
- `passes`: a function from a candidate to `Bool`.
- `candidates::AbstractVector`: the candidates, from least to most accurate. Not empty.

# Returns
- The chosen candidate.
"""
function _smallest_passing(passes, candidates::AbstractVector)
    low, high = firstindex(candidates), lastindex(candidates)
    while low < high
        middle = (low + high) ÷ 2
        if passes(candidates[middle])
            high = middle
        else
            low = middle + 1
        end
    end
    return candidates[low]
end

"""
    _fastest_candidate(time_of, candidates, start; clearly_slower = 1.3)
        -> Union{eltype(candidates), Nothing}

Walk a list of candidates in both directions from `start`, looking for the fastest. The time
per call against the grid is one broad hump with a jagged top (paper §5.2, Fig. 2), so a
slowdown or two does not mean the best has passed. In each direction the walk stops only
after two candidates in a row that are clearly slower than the fastest found so far, or at
the end of the list. Until a usable candidate turns up, the walk keeps going toward the end
of the list, because later candidates are the more accurate ones.

# Arguments
- `time_of`: a function from a candidate to its time, with `Inf` marking a candidate that
  can't be used. Called at most once per candidate.
- `candidates::AbstractVector`: the candidates, from least to most accurate.
- `start::Integer`: the index to begin at.

# Keywords
- `clearly_slower::Real = 1.3`: how many times slower than the fastest so far a candidate
  must be to count as clearly slower. Dips near the top of the jagged curves in Fig. 2 are
  about this size, and a single larger one doesn't stop the walk, since that takes two in a
  row. Away from the top the time soon grows past this, which keeps the walk short.

# Returns
- The fastest candidate found, or `nothing` if every candidate tried returned `Inf`.
"""
function _fastest_candidate(
    time_of, candidates::AbstractVector, start::Integer; clearly_slower::Real = 1.3,
)
    best = start
    best_time = time_of(candidates[start])
    for direction in (1, -1)
        index = start
        clearly_slower_in_a_row = 0
        while clearly_slower_in_a_row < 2 && index + direction in eachindex(candidates)
            index += direction
            candidate_time = time_of(candidates[index])
            if candidate_time < best_time
                best, best_time = index, candidate_time
                clearly_slower_in_a_row = 0
            elseif candidate_time < clearly_slower * best_time ||
                   (direction > 0 && !isfinite(best_time))
                clearly_slower_in_a_row = 0
            else
                clearly_slower_in_a_row += 1
            end
        end
    end
    return isfinite(best_time) ? candidates[best] : nothing
end

"""
    _fastest_taking_turns(time_of, candidates; rounds = 2) -> eltype(candidates)

Choose among close contenders fairly. Machines speed up and slow down over time, so each
candidate is timed in turn, round after round, and keeps its fastest time. Taking turns
gives them all the same conditions.

# Arguments
- `time_of`: a function from a candidate to its time, with `Inf` marking one that can't be
  used.
- `candidates::AbstractVector`: the contenders. At least one must be usable.

# Keywords
- `rounds::Integer = 2`: how many times each candidate is timed. A second round gives a
  candidate caught by a slow moment another chance, and more would spend a long time on
  candidates that are nearly tied anyway.

# Returns
- The candidate with the smallest time.
"""
function _fastest_taking_turns(time_of, candidates::AbstractVector; rounds::Integer = 2)
    fastest_times = fill(Inf, length(candidates))
    for _ in 1:rounds, position in eachindex(candidates)
        candidate_time = time_of(candidates[position])
        fastest_times[position] = min(fastest_times[position], candidate_time)
    end
    return candidates[argmin(fastest_times)]
end

"""
    _tighten_until_met(
        attempt, target, fallback; most_tightenings = 3, factor = 3,
    ) -> result

Run `attempt` with a tighter target each time it fails, then give up and return `fallback`
with a warning. Settings that met their own targets can still miss the tolerance overall,
because the error shifts a little with `Σ/σ`.

# Arguments
- `attempt`: a function from a target (`Float64`) to a result, or `nothing` on failure.
- `target::Float64`: the first target.
- `fallback`: what to return if every attempt fails.

# Keywords
- `most_tightenings::Integer = 3`: how many times to tighten before giving up. Each one
  repeats the whole search. With the default factor, three bring each search's share of the
  tolerance down to about the accuracy of the reference itself, so going further could not
  be measured.
- `factor::Real = 3`: what each tightening divides the target by. The error falls steeply as
  a setting grows, so this moves each setting by only about 5 %, which keeps the extra cost
  small when one tightening is enough.

# Returns
- The first successful result, or `fallback`.
"""
function _tighten_until_met(
    attempt, target::Float64, fallback; most_tightenings::Integer = 3, factor::Real = 3,
)
    for tightening in 0:most_tightenings
        result = attempt(target / factor^tightening)
        result === nothing || return result
    end
    @warn "no settings met the tolerance, however far the targets were tightened; " *
          "returning the very accurate reference settings, which are correct but slow"
    return fallback
end

"""
    _median_call_time(
        call; lead_in_seconds = 0.1, least_calls = 3, least_seconds = 0.2,
    ) -> Float64

The time one call takes, in seconds, measured fairly. A short untimed lead-in lets the
machine settle into this kind of work, because switching work changes how fast a processor
or GPU runs. Then the call is timed repeatedly and the median (the middle time) is kept, so
an occasional slow call doesn't move it.

# Arguments
- `call`: a function of no arguments, the call to time.

# Keywords
- `lead_in_seconds::Real = 0.1`: the length of the lead-in, which always includes at least
  one call. This is enough to settle, and small beside the time spent setting up a grid.
- `least_calls::Integer = 3`: the fewest timed calls. Three is the fewest for which a median
  can ignore a slow call, and it is where long calls stop, which keeps timing a large
  problem down to a few calls per grid.
- `least_seconds::Real = 0.2`: the least time spent on timed calls. A call of a millisecond
  or so needs many repeats for a steady median, and this gives it about a hundred.

# Returns
- `Float64`: the median time in seconds.
"""
function _median_call_time(
    call; lead_in_seconds::Real = 0.1, least_calls::Integer = 3, least_seconds::Real = 0.2,
)
    lead_in_start = time()
    call()
    while time() - lead_in_start < lead_in_seconds
        call()
    end
    times = Float64[]
    timing_start = time()
    while length(times) < least_calls || time() - timing_start < least_seconds
        push!(times, @elapsed call())
    end
    return partialsort!(times, (length(times) + 1) ÷ 2)
end

"""
    _fft_friendly_grids(L, T; available_bytes = 0.8 * Sys.free_memory())
        -> Vector{NTuple{3, Int32}}

The grids worth trying in a box: each size is made only of the factors 2, 3, 5 and 7, where
FFTs are fastest, with equal spacing along all axes (paper §3) and buffers that fit in
memory. Equal spacing is checked in double precision, so the grids are accepted in either
precision.

# Arguments
- `L::NTuple{3, Float64}`: the domain lengths.
- `T::Type{<:AbstractFloat}`: the precision the buffers would be in.

# Keywords
- `available_bytes::Real = 0.8 * Sys.free_memory()`: the memory the grid's buffers may take.
  The remaining fifth leaves room for the particle buffers, the FFT plans, and whatever else
  the machine is running. 20% is just a guess.

# Returns
- `Vector{NTuple{3, Int32}}`: the grids, from coarse to fine. Empty if the sides of the box
  share no usable spacing.
"""
function _fft_friendly_grids(
    L::NTuple{3, Float64}, ::Type{T}; available_bytes::Real = 0.8 * Sys.free_memory(),
) where {T}
    smallest_side = minimum(L)
    grids = NTuple{3, Int32}[]
    M = 2
    while true
        sizes = round.(Int, M .* L ./ smallest_side)
        _grid_buffer_bytes(T, sizes) ≤ available_bytes || return grids
        if all(M_i -> nextprod((2, 3, 5, 7), M_i) == M_i, sizes)
            grid = Int32.(sizes)
            _has_equal_grid_spacing(L, grid) && push!(grids, grid)
        end
        M += 1
    end
end

"""
    _index_with_spacing_at_most(L, grids, spacing) -> Int

The position of the coarsest grid whose spacing is no larger than `spacing`, or of the
finest grid if none is.

# Arguments
- `L::NTuple{3}`: the domain lengths.
- `grids::Vector{NTuple{3, Int32}}`: the grids, from coarse to fine. Not empty.
- `spacing::Real`: the largest acceptable spacing.

# Returns
- `Int`: the position in `grids`.
"""
function _index_with_spacing_at_most(
    L::NTuple{3}, grids::Vector{NTuple{3, Int32}}, spacing::Real,
)
    return something(findfirst(grid -> L[1] / grid[1] ≤ spacing, grids), lastindex(grids))
end

"""
    _unit_sphere_kernel_width(T) -> T

The Gaussian width of a unit sphere, `σ = a/√π` with `a = 1` (paper §2).

# Arguments
- `T::Type{<:AbstractFloat}`: the precision.

# Returns
- `T`: the width `σ`.
"""
_unit_sphere_kernel_width(::Type{T}) where {T} = one(T) / sqrt(T(π))

"""
    _generous_ratios(target_error) -> NamedTuple

Settings accurate enough that their own error stays near `target_error`, for use wherever a
calibration measurement needs one source of error to be negligible while it studies another:
the reference velocities, and the settings not being searched in each stage of `calibrate`.
They play the part of the very accurate runs in Su & Keaveny (2024): `R_c = L` in §5.1 and
`Σ/Δx = 2.0` in §5.2. Those fixed values would also work, but they cost up to ten times as
much per trial, while these scale with the accuracy actually requested.

These are not the method's settings, and these formulas are not from the paper. Each one
estimates how quickly one source of error decays, with a constant fitted to measurements on
suspensions of 1,000 to 64,000 particles:

- Grid resolution. Spreading and interpolation together damp a Fourier mode by
  `exp(-k²Σ²)`, which at the grid's highest mode `k = π/Δx` is `exp(-π²(Σ/Δx)²)`. The
  measured error stays below about 2.5 times that once `Σ/σ ≥ 1.5`, so
  `Σ/Δx = √(ln(2.5/e))/π`.
- Gaussian cut-off. What matters is how many Gaussian widths the stencil covers,
  `M_G/(Σ/Δx)`, not `M_G` alone. With `w` widths on each side the measured error is about
  `25 exp(-w²/2)`, so `w = √(2 ln(25/e))`. `_generous_support` then rounds `M_G` up to an
  odd number, because the stencil is anchored at the nearest grid point and an even `M_G`
  adds little over the odd number below it.
- Pair cutoff. The correction of equation (31) decays as `exp(-r²/4Σ²)`, which at
  `r = λΣ` is `exp(-λ²/4)`, so `λ = 2√(ln(1/e))`. This is close to measured values at tight
  tolerances and deliberately larger than them at loose ones.

An estimate that proved too optimistic can't silently spoil a calibration: the reference is
checked at two Gaussian widths, and every returned setting has its error measured against
that reference.

# Arguments
- `target_error::Float64`: the error the settings should stay near, well below the requested
  tolerance (see `generous_margin` in `CalibrationProblem`).

# Returns
- `NamedTuple` with `kernel_resolution` (`Σ/Δx`), `widths_covered` (`M_G/(Σ/Δx)`), and
  `cutoff_ratio` (`λ_ε = R_c/Σ`), all `Float64`.
"""
function _generous_ratios(target_error::Float64)
    return (;
        kernel_resolution = sqrt(log(2.5 / target_error)) / π,
        widths_covered = 2 * sqrt(2 * log(25 / target_error)),
        cutoff_ratio = 2 * sqrt(log(1 / target_error)),
    )
end

"""
    _generous_support(kernel_resolution, widths_covered) -> Int

The stencil width `M_G` covering at least `widths_covered` Gaussian widths at this grid
resolution, rounded up to an odd number (see `_generous_ratios`).

# Arguments
- `kernel_resolution::Real`: the grid resolution `Σ/Δx`.
- `widths_covered::Real`: the Gaussian widths to cover, `M_G/(Σ/Δx)`.

# Returns
- `Int`: the odd stencil width.
"""
function _generous_support(kernel_resolution::Real, widths_covered::Real)
    M_G = ceil(Int, widths_covered * kernel_resolution)
    return isodd(M_G) ? M_G : M_G + 1
end

"""
    _balanced_grid_spacing(
        L, N, kernel_resolution, cutoff_ratio; pair_to_grid_point_cost = 30,
    ) -> Float64

A starting guess for the fastest grid spacing, where grid work and pair work should balance.
This is an ansatz and not from the paper, whose §5.2 asks in words for such a
balance. Grid work grows like `V/Δx³`. Pair work grows like the number of pairs within
`R_c = λ_ε(Σ/Δx)Δx`, which is `(2π/3)(N²/V)R_c³`. Their sum is smallest where they are
equal, which gives

    Δx = (V/N)^(1/3) / (c^(1/6) √(λ_ε Σ/Δx)),

where `c` is `2π/3` times the cost of a pair relative to a grid point, which depends on the
machine.

# Arguments
- `L::NTuple{3}`: the domain lengths.
- `N::Integer`: the number of particles.
- `kernel_resolution::Real`: the grid resolution `Σ/Δx`.
- `cutoff_ratio::Real`: the reach of the pair correction `λ_ε = R_c/Σ`.

# Keywords
- `pair_to_grid_point_cost::Real = 30`: the cost `c`. This is a rounded choice and not a
  measurement: on the processor we measured, the value that fits lies between about 3 and
  30. Because of the sixth root, being wrong by a factor of ten moves the guess by less than
  a factor of 1.5, and the guess only decides where the search for the fastest grid begins.
  So, 30 should be close enough for the purposes of calibration. It beats starting from the
  coarsest option.

# Returns
- `Float64`: the grid spacing.
"""
function _balanced_grid_spacing(
    L::NTuple{3}, N::Integer, kernel_resolution::Real, cutoff_ratio::Real;
    pair_to_grid_point_cost::Real = 30,
)
    distance_between_particles = cbrt(prod(Float64, L) / N)
    return distance_between_particles /
           (pair_to_grid_point_cost^(1 / 6) * sqrt(cutoff_ratio * kernel_resolution))
end

"""
    _measuring_width(L, N, generous, smallest_spacing) -> Float64

The Gaussian width `Σ` at which accuracy is measured: the one that balances grid work and
pair work for the generous settings, which keeps the trials cheap. It is at least `2σ`, so
that the correction is really at work and the values found also hold for wider Gaussians,
and small enough that the generous cutoff fits in half the box. If the box forbids anything
wider than `σ`, it is `σ`. It is also wide enough that its grid need not be finer than
`smallest_spacing`, which is how a shortage of memory is handled.

# Arguments
- `L::NTuple{3, Float64}`: the domain lengths.
- `N::Integer`: the number of particles.
- `generous::NamedTuple`: as from `_generous_ratios`.
- `smallest_spacing::Float64`: the smallest grid spacing the measurements may use.

# Returns
- `Float64`: the width `Σ`.
"""
function _measuring_width(
    L::NTuple{3, Float64}, N::Integer, generous::NamedTuple, smallest_spacing::Float64,
)
    σ = _unit_sphere_kernel_width(Float64)
    (; kernel_resolution, cutoff_ratio) = generous
    balanced = kernel_resolution *
               _balanced_grid_spacing(L, N, kernel_resolution, cutoff_ratio)
    widest_that_fits_box = minimum(L) / (2 * cutoff_ratio)
    width = max(σ, min(max(2σ, balanced), widest_that_fits_box))
    return max(width, kernel_resolution * smallest_spacing)
end

"""
    _settings_on_grid(L, N, grid, ratios; worthwhile_widening = 1) -> NamedTuple

Turn accuracy ratios into the settings `FFCMConfig` takes on a given grid (paper §5): the
Gaussian width `Σ = (Σ/Δx)Δx` and the cutoff `R_c = λ_ε Σ`. Too little widening leaves the
Gaussian at `Σ = σ`, where the pair correction vanishes (standard FCM, §2), so the cutoff
only sizes the cells: about one particle per cell, and never beyond half the box.
`_settings_fit` says whether the result fits the box and the grid.

# Arguments
- `L::NTuple{3, T}`: the domain lengths.
- `N::Integer`: the number of particles.
- `grid::NTuple{3, Int32}`: the grid, with equal spacing along all axes.
- `ratios::NamedTuple`: `kernel_resolution` (`Σ/Δx`), `M_G`, and `cutoff_ratio` (`λ_ε`).

# Keywords
- `worthwhile_widening::Real = 1`: the `Σ/σ` below which the Gaussian stays at `σ`. The
  default only keeps it from being narrower than the particle's.

# Returns
- `NamedTuple` of `FFCMConfig` keywords: `kernel_widths_ratio::T`, `num_grid_points`, `M_G`,
  and `R_c::T`.
"""
function _settings_on_grid(
    L::NTuple{3, T}, N::Integer, grid::NTuple{3, Int32}, ratios::NamedTuple;
    worthwhile_widening::Real = 1,
) where {T}
    σ = _unit_sphere_kernel_width(T)
    widened = T(ratios.kernel_resolution) * (L[1] / grid[1])
    Σ = widened < T(worthwhile_widening) * σ ? σ : widened
    R_c = Σ == σ ? min(cbrt(prod(L) / T(N)), minimum(L) / T(2)) : T(ratios.cutoff_ratio) * Σ
    return (;
        kernel_widths_ratio = Σ / σ, num_grid_points = grid, M_G = Int(ratios.M_G), R_c,
    )
end

"""
    _settings_fit(settings, L) -> Bool

Whether `FFCMConfig` would accept these settings in this box: the cutoff fits the box and
the stencil fits the grid.

# Arguments
- `settings::NamedTuple`: as from `_settings_on_grid`.
- `L::NTuple{3}`: the domain lengths, in the precision of `settings`.

# Returns
- `Bool`: whether they fit.
"""
function _settings_fit(settings::NamedTuple, L::NTuple{3})
    return _cutoff_fits_box(settings.R_c, L) &&
           _support_fits_grid(settings.M_G, settings.num_grid_points)
end
