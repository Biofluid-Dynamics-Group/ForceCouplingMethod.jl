# The benchmark problems, shared by the benchmark suite (benchmarks.jl) and the reports
# (report.jl): the suspensions the paper times in §5.2 (Su & Keaveny 2024, Fig. 2 and
# Table 3).
#
# Each problem is a periodic cubic box of side 400 holding unit spheres (a/L = 1/400) at
# one of five volume fractions, placed at random without overlapping, with a random force
# on each sphere. The settings are the ones §5.2 takes from Tables 1–2 for an error of
# 1e-4: Σ/Δx = 1.00, M_G = 10 and λ_ε = R_c/Σ = 5.5. The remaining setting, Σ/σ, is
# Table 3's fastest value for each volume fraction. With Σ = Δx it fixes the grid M, as
# Σ/σ = 400/(M σ) with σ = 1/√π. The paper does not print its grids, and Table 3 gives
# Σ/σ to two figures: the grids for 0.2 % and 0.8 % are the only ones reproducing both of
# Table 3's values, and for the other three, where the rounding allows nearby grids too
# (489 to 525 for the densest), these are the round sizes. For the densest suspension
# Table 3 prints R_c/σ = 8.1, a little above λ_ε Σ/σ = 7.8; we follow λ_ε = 5.5, as §5.2
# states.

using Random: Xoshiro

const BOX = 400.0
const KERNEL_RESOLUTION = 1.0  # Σ/Δx
const STENCIL_SUPPORT = 10     # M_G
const CUTOFF_RATIO = 5.5       # λ_ε = R_c/Σ

# Each problem is named by its volume fraction in per cent, as in Table 3.
const PROBLEMS = [
    (name = "0.05", volume_fraction = 0.0005, grid = 120),
    (name = "0.2", volume_fraction = 0.002, grid = 140),
    (name = "0.8", volume_fraction = 0.008, grid = 210),
    (name = "3.2", volume_fraction = 0.032, grid = 400),
    (name = "12.8", volume_fraction = 0.128, grid = 500),
]
const PROBLEM_NAMES = [problem.name for problem in PROBLEMS]

# The problem `name` with its derived settings: the number of spheres, rounded down as the
# paper does (its Table 1 suspension of 8 % at a/L = 1/150 has 64,457 spheres), the
# kernel-width ratio Σ/σ and the cutoff R_c.
function benchmark_problem(name::AbstractString)
    index = findfirst(problem -> problem.name == name, PROBLEMS)
    index === nothing && throw(ArgumentError(
        "unknown benchmark problem $(name); the problems are named by their volume " *
        "fraction in per cent: " * join(PROBLEM_NAMES, ", "),
    ))
    problem = PROBLEMS[index]
    σ = 1 / sqrt(π)  # Gaussian width of a unit sphere (paper §2)
    Σ = KERNEL_RESOLUTION * BOX / problem.grid
    particles = floor(Int, problem.volume_fraction * BOX^3 / (4π / 3))
    return (; problem..., particles, kernel_widths_ratio = Σ / σ, R_c = CUTOFF_RATIO * Σ, σ)
end

# The keyword arguments of `FFCMConfig{T}` that set up `problem`.
function problem_keywords(::Type{T}, problem) where {T}
    return (
        L = (T(BOX), T(BOX), T(BOX)),
        N = problem.particles,
        kernel_widths_ratio = T(problem.kernel_widths_ratio),
        num_grid_points = ntuple(_ -> Int32(problem.grid), 3),
        M_G = STENCIL_SUPPORT,
        R_c = T(problem.R_c),
        viscosity = T(1),
    )
end

# A one-line description of `problem` for the reports.
function describe_problem(problem)
    particles = replace(string(problem.particles), r"(?<=\d)(?=(\d{3})+$)" => ",")
    Σ_over_σ = round(problem.kernel_widths_ratio; digits = 2)
    R_c_over_σ = round(problem.R_c / problem.σ; digits = 1)
    return "volume fraction $(problem.name) %: $(problem.grid)³ grid, $(particles) " *
           "particles, Σ/σ = $(Σ_over_σ), R_c/σ = $(R_c_over_σ)"
end

# Positions of `count` unit spheres placed at random in a periodic cube of side `box`,
# none overlapping another. The paper's suspensions have no overlaps (§5.2), and spheres
# placed independently at random would overlap often: at 12.8 % about two in three would
# touch a neighbour. Each sphere is placed at a uniformly random point and kept only if
# no sphere placed before it lies closer than one diameter (random sequential addition).
# Dividing the box into cells at least one diameter wide means only the 27 cells around
# a new point need checking.
function non_overlapping_positions(rng, box, count)
    cells_per_side = max(floor(Int, box / 2), 1)
    cell_width = box / cells_per_side
    first_in_cell = zeros(Int32, cells_per_side, cells_per_side, cells_per_side)
    next_in_cell = zeros(Int32, count)
    positions = Matrix{Float64}(undef, 3, count)
    placed = 0
    attempts = 0
    while placed < count
        attempts += 1
        attempts ≤ 1000 * count || error(
            "could not place $(count) non-overlapping spheres in a box of side $(box)",
        )
        x = (box * rand(rng), box * rand(rng), box * rand(rng))
        cell = ntuple(i -> min(floor(Int, x[i] / cell_width), cells_per_side - 1), 3)
        overlaps = false
        for dz in -1:1, dy in -1:1, dx in -1:1
            neighbour = mod.(cell .+ (dx, dy, dz), cells_per_side) .+ 1
            other = first_in_cell[neighbour...]
            while other != 0
                distance² = 0.0
                for i in 1:3
                    d = x[i] - positions[i, other]
                    distance² += (d - box * round(d / box))^2  # nearest periodic image
                end
                overlaps = distance² < 4.0  # closer than one diameter
                overlaps && break
                other = next_in_cell[other]
            end
            overlaps && break
        end
        overlaps && continue
        placed += 1
        positions[:, placed] .= x
        next_in_cell[placed] = first_in_cell[(cell .+ 1)...]
        first_in_cell[(cell .+ 1)...] = placed
    end
    return positions
end

# The suspension of `problem` in precision `T`: non-overlapping random positions and a
# force of order one on each sphere, with every component uniform in [-1, 1]. A fixed seed
# means every run, and both precisions, time the same particles.
function paper_suspension(::Type{T}, problem) where {T}
    rng = Xoshiro(2024)
    Y = T.(non_overlapping_positions(rng, BOX, problem.particles))
    F = T.(2 .* rand(rng, 3, problem.particles) .- 1)
    return Y, F
end
