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
        kernel_resolution::Float64,
        widths_covered::Float64,
        cutoff_ratio::Float64,
    }
    measuring_width::Float64
    exact_velocities::Matrix{Float64}
    reference_uncertainty::Float64
    reference_settings::@NamedTuple{
        kernel_widths_ratio::T,
        num_grid_points::NTuple{3, Int32},
        M_G::Int,
        R_c::T,
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
- `ArgumentError`: if no equal-spacing grid fits in memory, if the generous settings don't
  fit the box and the memory, or if the two widths disagree (see
  `_check_reference_agreement`).
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
    # The second reference grid is finer still, so the first has to leave it room in memory.
    finest_spacing = L_double[1] / grids_double[end][1]
    smallest_spacing = finest_spacing / finer_reference_spacing
    measuring_width = _measuring_width(L_double, N, generous, smallest_spacing)
    ratios = _ratios_with_generous_stencil(
        generous, generous.kernel_resolution, generous.cutoff_ratio,
    )
    spacing = measuring_width / generous.kernel_resolution
    coarser_grid = _grid_with_spacing_at_most(L_double, grids_double, spacing)
    finer_grid = _grid_with_spacing_at_most(
        L_double, grids_double, finer_reference_spacing * spacing,
    )
    coarser = _settings_on_grid(L_double, N, coarser_grid, ratios)
    finer = _settings_on_grid(L_double, N, finer_grid, ratios)
    reference_settings = _settings_on_grid(L, N, finer_grid, ratios)
    if coarser === nothing || finer === nothing || reference_settings === nothing
        throw(ArgumentError(
            "there is no room for the very accurate reference calculation: its grid or " *
            "its cutoff doesn't fit the available memory and a box of size $(L)",
        ))
    end
    Y_double, F_double = Matrix{Float64}(Y), Matrix{Float64}(F)
    planning = (; fft_planning = :estimate, fft_threads)
    V_coarser = _velocities_with(L_double, Y_double, F_double, coarser; planning...)
    V_finer = _velocities_with(L_double, Y_double, F_double, finer; planning...)
    reference_uncertainty = Float64(reference_agreement * tolerance)
    _check_reference_agreement(V_coarser, V_finer, reference_uncertainty)
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
    _velocities_with(L, Y, F, settings; fft_planning, fft_threads) -> Matrix

One application of the mobility operator with the given settings. The configuration is freed
when this returns, which matters when many large ones are tried in a row. The backend is
chosen at run time, so the compiler doesn't know the configuration's exact type here and
`mobility!` is dispatched once per trial. Everything inside it is compiled for the exact
types.

# Arguments
- `L::NTuple{3, T}`: the domain lengths.
- `Y::Matrix{T}`, `F::Matrix{T}`: the `3xN` positions and forces.
- `settings::NamedTuple`: `FFCMConfig` keywords, as from `_settings_on_grid`.

# Keywords
- `fft_planning::Symbol`, `fft_threads::Integer`: as for `FFCMConfig`.

# Returns
- `Matrix{T}`: the `3xN` velocities, at unit viscosity. Relative errors and times don't
  depend on the viscosity.
"""
function _velocities_with(
    L::NTuple{3, T},
    Y::Matrix{T},
    F::Matrix{T},
    settings::NamedTuple;
    fft_planning::Symbol,
    fft_threads::Integer,
) where {T}
    config = FFCMConfig{T}(;
        L, N = size(Y, 2), viscosity = one(T), fft_planning, fft_threads, settings...,
    )
    return mobility!(similar(Y), config, Y, F)
end

"""
    _check_reference_agreement(V_coarser, V_finer, largest_disagreement) -> nothing

Compare the very accurate velocities computed at two Gaussian widths. They must agree
closely, since the exact answer doesn't depend on the width (paper §3).

# Arguments
- `V_coarser::AbstractMatrix`, `V_finer::AbstractMatrix`: the two `3xN` velocity matrices.
- `largest_disagreement::Float64`: the largest relative error allowed between them.

# Returns
- `nothing`.

# Throws
- `ArgumentError`: if they disagree, or if either is not finite, as when two particles share
  a position and the correction between them is undefined.
"""
function _check_reference_agreement(
    V_coarser::AbstractMatrix,
    V_finer::AbstractMatrix,
    largest_disagreement::Float64,
)
    disagreement = _relative_velocity_error(V_coarser, V_finer)
    disagreement ≤ largest_disagreement && return nothing
    throw(ArgumentError(
        "the very accurate velocities computed at two Gaussian widths differ by " *
        "$(disagreement), more than the $(largest_disagreement) allowed, although the " *
        "exact answer doesn't depend on the width. Check that no two particles share a " *
        "position",
    ))
end

"""
    _relative_velocity_error(V, exact) -> Float64

The error measure of Su & Keaveny (2024), §5.1 equation (69): `|V_n - exact_n| / |exact_n|`
for each particle `n`, averaged over the particles. A particle whose exact velocity is zero
has no relative error and is left out of the average. Exact here just means computed with a
very accurate method, it is not actually exact.

# Arguments
- `V::AbstractMatrix`: the `3xN` velocities being judged, in any precision.
- `exact::AbstractMatrix`: the `3xN` velocities taken as exact. At least one must be
  nonzero.

# Returns
- `Float64`: the mean relative error.

# Throws
- `ArgumentError`: if every exact velocity is zero, so that the error is undefined.
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
    moving_particles > 0 || throw(ArgumentError(
        "the relative velocity error is undefined when no particle moves",
    ))
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

# Throws
- `ArgumentError`: if `candidates` is empty.
"""
function _smallest_passing(passes, candidates::AbstractVector)
    isempty(candidates) && throw(ArgumentError("there are no candidates to search"))
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
    _fastest_candidate(time_of, count, start; clearly_slower = 1.3) -> Union{Int, Nothing}

Walk a list of candidates in both directions from `start`, looking for the fastest. The time
per call against the grid is one broad hump with a jagged top (paper §5.2, Fig. 2), so a
slowdown or two does not mean the best has passed. In each direction the walk stops only
after two candidates in a row that are clearly slower than the fastest found so far, or at
the end of the list.

# Arguments
- `time_of`: a function from a candidate's index to its time, with `Inf` marking a candidate
  that can't be used. Called at most once per index.
- `count::Integer`: the number of candidates, indexed `1:count`.
- `start::Integer`: the index to begin at.

# Keywords
- `clearly_slower::Real = 1.3`: how many times slower than the fastest so far a candidate
  must be to count as clearly slower. Dips near the top of the jagged curves in Fig. 2 are
  about this size, and a single larger one doesn't stop the walk, since that takes two in a
  row. Away from the top the time soon grows past this, which keeps the walk short.

# Returns
- `Int`: the index of the fastest candidate found, or `nothing` if every candidate tried
  returned `Inf`.
"""
function _fastest_candidate(
    time_of,
    count::Integer,
    start::Integer;
    clearly_slower::Real = 1.3,
)
    best = start
    best_time = time_of(start)
    for direction in (1, -1)
        index = start
        clearly_slower_in_a_row = 0
        while clearly_slower_in_a_row < 2 && 1 ≤ index + direction ≤ count
            index += direction
            candidate_time = time_of(index)
            if candidate_time < best_time
                best, best_time = index, candidate_time
                clearly_slower_in_a_row = 0
            elseif candidate_time ≥ clearly_slower * best_time
                clearly_slower_in_a_row += 1
            else
                clearly_slower_in_a_row = 0
            end
        end
    end
    return isfinite(best_time) ? best : nothing
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
    attempt,
    target::Float64,
    fallback;
    most_tightenings::Integer = 3,
    factor::Real = 3,
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
    call;
    lead_in_seconds::Real = 0.1,
    least_calls::Integer = 3,
    least_seconds::Real = 0.2,
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
    L::NTuple{3, Float64},
    ::Type{T};
    available_bytes::Real = 0.8 * Sys.free_memory(),
) where {T}
    smallest_side = minimum(L)
    grids = NTuple{3, Int32}[]
    M = 2
    while true
        sizes = round.(Int, M .* L ./ smallest_side)
        _grid_buffer_bytes(T, sizes) ≤ available_bytes || return grids
        if all(_is_fft_friendly, sizes)
            grid = Int32.(sizes)
            _has_equal_grid_spacing(L, grid) && push!(grids, grid)
        end
        M += 1
    end
end

"""
    _is_fft_friendly(M) -> Bool

Whether a grid size is made only of the factors 2, 3, 5 and 7, where FFTs are fastest.

# Arguments
- `M::Integer`: the number of grid points along an axis.

# Returns
- `Bool`: whether the size is FFT-friendly. Zero is not.
"""
function _is_fft_friendly(M::Integer)
    remainder = M
    for factor in (2, 3, 5, 7)
        while remainder > 0 && remainder % factor == 0
            remainder ÷= factor
        end
    end
    return remainder == 1
end

"""
    _grid_with_spacing_at_most(L, grids, spacing) -> NTuple{3, Int32}

The coarsest grid whose spacing is no larger than `spacing`, or the finest grid if none is.

# Arguments
- `L::NTuple{3}`: the domain lengths.
- `grids::Vector{NTuple{3, Int32}}`: the grids, from coarse to fine. Not empty.
- `spacing::Real`: the largest acceptable spacing.

# Returns
- `NTuple{3, Int32}`: the grid.
"""
function _grid_with_spacing_at_most(
    L::NTuple{3},
    grids::Vector{NTuple{3, Int32}},
    spacing::Real,
)
    index = findfirst(grid -> Float64(L[1]) / grid[1] ≤ spacing, grids)
    return index === nothing ? grids[end] : grids[index]
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
    _ratios_with_generous_stencil(generous, kernel_resolution, cutoff_ratio) -> NamedTuple

Accuracy ratios whose stencil covers a generous number of Gaussian widths at this grid
resolution, so that cutting off the Gaussian adds no noticeable error. The stencil has to
grow with `Σ/Δx`: at a fixed `M_G`, the error turns back up as `Σ/Δx` grows.

# Arguments
- `generous::NamedTuple`: as from `_generous_ratios`.
- `kernel_resolution::Real`: the grid resolution `Σ/Δx`.
- `cutoff_ratio::Real`: the reach of the pair correction `λ_ε = R_c/Σ`.

# Returns
- `NamedTuple` with `kernel_resolution`, `M_G`, and `cutoff_ratio`, as `_settings_on_grid`
  takes.
"""
function _ratios_with_generous_stencil(
    generous::NamedTuple,
    kernel_resolution::Real,
    cutoff_ratio::Real,
)
    M_G = _generous_support(kernel_resolution, generous.widths_covered)
    return (; kernel_resolution, M_G, cutoff_ratio)
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

where `c` is the cost of a pair relative to a grid point, which depends on the machine.

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
    L::NTuple{3},
    N::Integer,
    kernel_resolution::Real,
    cutoff_ratio::Real;
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
    L::NTuple{3, Float64},
    N::Integer,
    generous::NamedTuple,
    smallest_spacing::Float64,
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
    _settings_on_grid(L, N, grid, ratios) -> Union{NamedTuple, Nothing}

Turn accuracy ratios into the settings `FFCMConfig` takes on a given grid (paper §5): the
Gaussian width `Σ = (Σ/Δx)Δx`, never narrower than the particle's `σ`, and the cutoff
`R_c = λ_ε Σ`. At `Σ = σ` the pair correction vanishes (standard FCM, §2), so the cutoff
only sizes the cells: about one particle per cell, and never beyond half the box.

# Arguments
- `L::NTuple{3, T}`: the domain lengths.
- `N::Integer`: the number of particles.
- `grid::NTuple{3, Int32}`: the grid, with equal spacing along all axes.
- `ratios::NamedTuple`: `kernel_resolution` (`Σ/Δx`), `M_G`, and `cutoff_ratio` (`λ_ε`).

# Returns
- `NamedTuple` of `FFCMConfig` keywords: `kernel_widths_ratio::T`, `num_grid_points`, `M_G`,
  and `R_c::T`. Or `nothing` if the cutoff doesn't fit the box or the stencil doesn't fit
  the grid.
"""
function _settings_on_grid(
    L::NTuple{3, T},
    N::Integer,
    grid::NTuple{3, Int32},
    ratios::NamedTuple,
) where {T}
    σ = _unit_sphere_kernel_width(T)
    Σ = max(σ, T(ratios.kernel_resolution) * (L[1] / grid[1]))
    R_c = Σ == σ ? min(cbrt(prod(L) / T(N)), minimum(L) / T(2)) : T(ratios.cutoff_ratio) * Σ
    (_cutoff_fits_box(R_c, L) && _support_fits_grid(ratios.M_G, grid)) || return nothing
    return (;
        kernel_widths_ratio = Σ / σ,
        num_grid_points = grid,
        M_G = Int(ratios.M_G),
        R_c,
    )
end

"""
    _candidate_settings(L, N, grid, ratios; worthwhile_widening = 1.25)
        -> Union{NamedTuple, Nothing}

The settings to try on a grid in the search for the fastest one: those of
`_settings_on_grid`, except that a Gaussian widened too little gives way to standard FCM.
So little widening saves little grid work, yet it needs pair work that standard FCM avoids.
The paper makes this point for dense suspensions (§5.2).

# Arguments
- `L`, `N`, `grid`, `ratios`: as for `_settings_on_grid`.

# Keywords
- `worthwhile_widening::Real = 1.25`: the `Σ/σ` below which standard FCM takes over. At
  about this widening, standard FCM on the same grid usually still meets the tolerance,
  because we measured its grid error to be 5 to 40 times smaller at the same spacing. Where
  it doesn't, the search for the fastest grid finds that out, since it measures every grid's
  error.

# Returns
- As for `_settings_on_grid`.
"""
function _candidate_settings(
    L::NTuple{3, T},
    N::Integer,
    grid::NTuple{3, Int32},
    ratios::NamedTuple;
    worthwhile_widening::Real = 1.25,
) where {T}
    σ = _unit_sphere_kernel_width(T)
    widening = ratios.kernel_resolution * (L[1] / grid[1]) / σ
    widening < worthwhile_widening || return _settings_on_grid(L, N, grid, ratios)
    # A zero resolution leaves the Gaussian at its narrowest, Σ = σ.
    return _settings_on_grid(L, N, grid, (; ratios..., kernel_resolution = 0.0))
end
