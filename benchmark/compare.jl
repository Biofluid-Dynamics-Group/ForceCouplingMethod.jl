# Compares two saved benchmark runs, measurement by measurement. From the repository root:
#
#     julia --project=benchmark benchmark/compare.jl <before folder> <after folder>
#
# where each folder is a run's benchmark/results/<label>/. A measurement counts as faster
# or slower only when its median changed by more than 5 % and the middle halves of the two
# runs' timings do not overlap; a smaller change is within the scatter between runs and is
# reported as "no real change". Compare runs from the same computer in a similar state:
# any differences between the two machine descriptions are listed first.

using BenchmarkTools
using Printf: @sprintf

include(joinpath(@__DIR__, "report.jl"))

const NOTICEABLE_CHANGE = 0.05

function verdict(before, after)
    change = after.median / before.median - 1
    change < -NOTICEABLE_CHANGE && after.high < before.low && return "faster"
    change > NOTICEABLE_CHANGE && after.low > before.high && return "slower"
    return "no real change"
end

# The entries of the two machine descriptions that differ, ignoring those that always
# change between runs (label, date, the machine state during the timings, and notes on
# skipped problems).
function machine_differences(before, after)
    differences = String[]
    for (key, value) in before
        ignored = key in ("Label", "Date", "Skipped") || startswith(key, "State of ")
        ignored && continue
        other = machine_value(after, key)
        other == value || push!(differences, "$(key): $(value) → $(other)")
    end
    return differences
end

length(ARGS) == 2 ||
    error("usage: julia --project=benchmark benchmark/compare.jl <before> <after>")
before_folder, after_folder = ARGS
before = BenchmarkTools.load(joinpath(before_folder, "timings.json"))[1]
after = BenchmarkTools.load(joinpath(after_folder, "timings.json"))[1]
before_machine = read_machine(joinpath(before_folder, "machine.txt"))
after_machine = read_machine(joinpath(after_folder, "machine.txt"))

println("# Before: ", machine_value(before_machine, "Label"), " — after: ",
        machine_value(after_machine, "Label"), "\n")
differences = machine_differences(before_machine, after_machine)
if !isempty(differences)
    println("The two runs differ in their machine or software, so some changes may come ",
            "from that rather than from the code:\n")
    foreach(difference -> println("- ", difference), differences)
    println()
end

println("| Backend | Problem | Precision | What is timed | Before | After | Change | ",
        "Verdict |")
println("|---|---|---|---|--:|--:|--:|---|")
counts = Dict("faster" => 0, "slower" => 0, "no real change" => 0)
for backend in ("cpu", "gpu")
    (haskey(before, backend) && haskey(after, backend)) || continue
    for problem in problems_in(before[backend]), (item, _) in vcat(CALL_ITEMS, SETUP_ITEMS)
        for precision in PRECISIONS
            leaf_keys = (backend, item, precision, problem.name)
            before_trial = find_trial(before, leaf_keys...)
            after_trial = find_trial(after, leaf_keys...)
            (before_trial === nothing || after_trial === nothing) && continue
            before_summary = timing_summary(before_trial)
            after_summary = timing_summary(after_trial)
            result = verdict(before_summary, after_summary)
            counts[result] += 1
            ratio = after_summary.median / before_summary.median
            println("| ", uppercase(backend), " | ", problem.name, " | ", precision, " | ",
                    item_label(backend, item), " | ",
                    format_duration(before_summary.median), " | ",
                    format_duration(after_summary.median), " | ",
                    @sprintf("%+.1f %%", 100 * (ratio - 1)), " | ", result, " |")
        end
    end
end
println("\n", counts["faster"], " faster, ", counts["slower"], " slower, ",
        counts["no real change"], " with no real change.")
