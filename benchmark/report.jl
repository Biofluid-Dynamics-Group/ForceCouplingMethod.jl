# Turns saved benchmark results into a readable Markdown report. run.jl uses it; on its
# own it shows a saved run again:
#
#     julia --project=benchmark benchmark/report.jl benchmark/results/<label>

using BenchmarkTools
using Printf: @sprintf
using Statistics: mean, median, quantile

isdefined(@__MODULE__, :benchmark_problem) || include(joinpath(@__DIR__, "problems.jl"))

# The timed items, in report order, with plain-English names: first the calls a simulation
# repeats, then the one-off setup and the FFT options. Items a run did not measure are left
# out of its report.
const CALL_ITEMS = [
    "mobility" => "Whole call (`mobility!`)",
    "mobility_gpu_data" => "Whole call, data already on the GPU",
    "copies" => "Copies between computer memory and the GPU",
    "wrap_positions" => "Step 1: wrap positions into the box",
    "assign_cells" => "Step 2: find each particle's cell",
    "sort_particles_by_cell" => "Step 3: sort particles by cell",
    "spread_forces" => "Step 4: spread forces onto the grid",
    "stokes_solve" => "Step 5: solve the Stokes equations on the grid (FFTs)",
    "interpolate_velocities" => "Step 6: interpolate velocities from the grid",
    "correct_velocities" => "Step 7: correct nearby pairs",
]
const SETUP_ITEMS = [
    "setup_without_saved_plans" => "Setup, first time (FFTW searches for FFT plans)",
    "setup_with_saved_plans" => "Setup, repeated (FFTW reuses its saved plans)",
    "setup" => "Setup from scratch (GPU memory and FFT plans)",
    "stokes_solve_quick_plans" => "Step 5 with quick FFT plans (`:estimate`, no search)",
    "stokes_solve_searched_plans" => "Step 5 with searched FFT plans (`:measure`, default)",
    "stokes_solve_threaded_plans" => "Step 5 with the FFTs split over all Julia threads",
]
# Steps 4–7: the grid and pair work, without building the cell list or copying data.
const GRID_AND_PAIR_STEPS = (
    "spread_forces", "stokes_solve", "interpolate_velocities", "correct_velocities",
)
const PRECISIONS = ("Float32", "Float64")

const READING_GUIDE = """
How to read the tables: each time is the **median** (the middle value) of 50 timed calls.
Before them the problem ran untimed for at least 50 calls and 10 seconds, and the timed
call itself for one more second. Setup times use 10 timed calls. The **middle half** is
the range that holds the middle half of the timed calls, from the 25th to the 75th
percentile: a narrow range means steady timings. **PTPS** is the paper's speed measure
(§5.2): the number of particles divided by the *average* time of one call. A dash means
the run did not measure that item.
"""

function item_label(backend, item)
    if backend == "gpu" && item == "mobility"
        return "Whole call, data in computer memory (what users see)"
    end
    for (key, label) in vcat(CALL_ITEMS, SETUP_ITEMS)
        key == item && return label
    end
    return item
end

# The measurement stored under `keys` in `results`, or `nothing` if the run lacks it.
function find_trial(results, keys...)
    group = results
    for key in keys
        haskey(group, key) || return nothing
        group = group[key]
    end
    return group
end

# The median of a measurement's timings, the range holding their middle half (from the
# 25th to the 75th percentile) and their average, in seconds. The median ignores an
# occasional slow call; the width of the middle half shows how much the timings scatter;
# the average is what the paper's PTPS uses.
function timing_summary(trial)
    seconds = trial.times ./ 1e9
    return (
        median = median(seconds),
        low = quantile(seconds, 0.25),
        high = quantile(seconds, 0.75),
        mean = mean(seconds),
        calls = length(seconds),
    )
end

# The unit, as (name, length in seconds), that writes `seconds` as a number below 1000.
function time_unit(seconds)
    seconds ≥ 0.9995 && return ("s", 1.0)
    seconds ≥ 0.9995e-3 && return ("ms", 1e-3)
    return ("µs", 1e-6)
end

# `x` with three significant figures, keeping trailing zeros (1.20, not 1.2).
function three_figures(x)
    x ≥ 999.5 && return string(round(Int, x))
    return rstrip(@sprintf("%#.3g", x), '.')
end

function format_duration(seconds)
    unit, unit_seconds = time_unit(seconds)
    return "$(three_figures(seconds / unit_seconds)) $(unit)"
end

# The range from `low` to `high`, in the unit that suits `reference`.
function format_range(low, high, reference)
    unit, unit_seconds = time_unit(reference)
    low_text = three_figures(low / unit_seconds)
    high_text = three_figures(high / unit_seconds)
    return "$(low_text)–$(high_text) $(unit)"
end

# A count rounded to three significant figures, with thousands separators.
function format_count(x)
    digits = string(round(Int, round(x; sigdigits = 3)))
    return replace(digits, r"(?<=\d)(?=(\d{3})+$)" => ",")
end

# The problems measured in `group` (one backend's results, keyed item / precision /
# problem), most dilute first.
function problems_in(group)
    names = unique(String(leaf_keys[3]) for (leaf_keys, _) in BenchmarkTools.leaves(group))
    sort!(names; by = name -> findfirst(==(name), PROBLEM_NAMES))
    return [benchmark_problem(name) for name in names]
end

# One Markdown table: for each item in `items` that `backend` measured on `problem`, the
# median and middle half of its timings in each precision.
function write_time_table(io, results, backend, problem, items)
    rows = Tuple{String, Vector{String}}[]
    for (item, _) in items
        cells = String[]
        for precision in PRECISIONS
            trial = find_trial(results, backend, item, precision, problem.name)
            if trial === nothing
                append!(cells, ("—", "—"))
            else
                summary = timing_summary(trial)
                push!(cells, format_duration(summary.median))
                push!(cells, format_range(summary.low, summary.high, summary.median))
            end
        end
        all(==("—"), cells) || push!(rows, (item_label(backend, item), cells))
    end
    isempty(rows) && return nothing

    println(io, "| What is timed | Float32 median | Float32 middle half | ",
            "Float64 median | Float64 middle half |")
    println(io, "|---|--:|--:|--:|--:|")
    for (label, cells) in rows
        println(io, "| ", label, " | ", join(cells, " | "), " |")
    end
    println(io)
    return nothing
end

# The paper's speed measure (§5.2): particles divided by the average time of one call, for
# the whole call and for steps 4–7 alone (the grid and pair work, without building the
# cell list or copying data).
function write_ptps_table(io, results, backend, problem)
    function ptps(precision, items)
        trials = [find_trial(results, backend, item, precision, problem.name)
                  for item in items]
        any(isnothing, trials) && return "—"
        seconds = sum(trial -> timing_summary(trial).mean, trials)
        return format_count(problem.particles / seconds)
    end
    rows = Pair{String, Vector{String}}[item_label(backend, "mobility") => ["mobility"]]
    backend == "gpu" &&
        push!(rows, item_label(backend, "mobility_gpu_data") => ["mobility_gpu_data"])
    push!(rows, "Steps 4–7 only (grid and pair work)" => collect(GRID_AND_PAIR_STEPS))

    println(io, "| PTPS (particles ÷ average time of one call) | Float32 | Float64 |")
    println(io, "|---|--:|--:|")
    for (label, items) in rows
        println(io, "| ", label, " | ", ptps("Float32", items), " | ",
                ptps("Float64", items), " |")
    end
    println(io)
    return nothing
end

# The machine state recorded during the run, as a table: one row per problem timed, with
# the state at the start and at the end of its timings.
function write_state_table(io, machine)
    states = filter(entry -> startswith(first(entry), "State of "), machine)
    isempty(states) && return nothing
    println(io, "## Machine state during the timings\n")
    println(io, "| Backend / problem / precision | At the start | At the end |")
    println(io, "|---|---|---|")
    for (key, value) in states
        endswith(key, " / start") || continue
        timed = chopsuffix(chopprefix(key, "State of "), " / start")
        println(io, "| ", timed, " | ", value, " | ",
                machine_value(machine, "State of $(timed) / end"), " |")
    end
    println(io)
    return nothing
end

# The full report: the machine description, a reading guide, then per backend and
# problem a table of the repeated calls, the PTPS, and a table of the setup and FFT
# options, and finally the machine state during the timings.
function write_report(io, results, machine)
    println(io, "# Benchmark report: ", machine_value(machine, "Label"), "\n")
    for (key, value) in machine
        (key == "Label" || startswith(key, "State of ")) && continue
        println(io, "- **", key, ":** ", value)
    end
    println(io, "\n", READING_GUIDE)
    for backend in ("cpu", "gpu")
        haskey(results, backend) || continue
        println(io, backend == "cpu" ? "## Processor (CPU)\n" : "## GPU\n")
        for problem in problems_in(results[backend])
            println(io, "### ", uppercasefirst(describe_problem(problem)), "\n")
            write_time_table(io, results, backend, problem, CALL_ITEMS)
            write_ptps_table(io, results, backend, problem)
            write_time_table(io, results, backend, problem, SETUP_ITEMS)
        end
    end
    write_state_table(io, machine)
    return nothing
end

# The machine description is stored as one "description: value" line per entry.
function write_machine(path, machine)
    open(path, "w") do io
        for (key, value) in machine
            println(io, key, ": ", value)
        end
    end
    return nothing
end

function read_machine(path)
    return [
        String(key) => String(value)
        for (key, value) in (split(line, ": "; limit = 2) for line in eachline(path))
    ]
end

function machine_value(machine, key)
    for (k, value) in machine
        k == key && return value
    end
    return "unknown"
end

# Every timing as one line of plain text, for plot_timings.py: run label, backend,
# precision, problem, timed item and its plain-English description, call number (in the
# order the calls were made) and duration in seconds.
function write_timings_csv(path, results, label)
    open(path, "w") do io
        println(io, "label,backend,precision,problem,item,description,call,seconds")
        for (leaf_keys, trial) in BenchmarkTools.leaves(results)
            backend, item, precision, problem = leaf_keys
            description = "\"" * item_label(backend, item) * "\""  # quoted: has commas
            for (call, nanoseconds) in enumerate(trial.times)
                fields = (
                    label, backend, precision, problem, item, description, call,
                    nanoseconds / 1e9,
                )
                println(io, join(fields, ","))
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        error("usage: julia --project=benchmark benchmark/report.jl <results folder>")
    results = BenchmarkTools.load(joinpath(ARGS[1], "timings.json"))[1]
    write_report(stdout, results, read_machine(joinpath(ARGS[1], "machine.txt")))
end
