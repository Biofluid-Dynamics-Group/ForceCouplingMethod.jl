# Runs the benchmark and writes a readable report. From the repository root:
#
#     julia -t auto --project=benchmark benchmark/run.jl [label]
#
# GPU timings run only when exactly one GPU is visible: choose it with
# CUDA_VISIBLE_DEVICES=<index>. The results go to benchmark/results/<label>/ (git-ignored,
# since timings belong to one machine), where the label defaults to the date and time. An
# existing folder is never overwritten. The folder holds:
#
#   report.md     the readable report, also printed at the end
#   machine.txt   the computer, software versions and machine state of the run
#   timings.json  every timing, read by compare.jl
#   timings.csv   every timing as plain text, read by plot_timings.py

using BenchmarkTools
using Dates: format, now

include(joinpath(@__DIR__, "benchmarks.jl"))
include(joinpath(@__DIR__, "report.jl"))

# `f()` as text, or "unknown" if it fails: some queries depend on the CUDA.jl version or
# on nvidia-smi being installed.
function query(f)
    try
        return string(f())
    catch
        return "unknown"
    end
end

# The visible GPU's identifier for nvidia-smi, which numbers GPUs differently from
# CUDA_VISIBLE_DEVICES.
visible_gpu_id() = "GPU-" * string(CUDA.uuid(CUDA.device()))

# A comma-separated reading of `fields` for the visible GPU.
function nvidia_smi(fields)
    device = visible_gpu_id()
    return readchomp(`nvidia-smi --id=$device --query-gpu=$fields --format=csv,noheader`)
end

# The visible GPU's temperature, clock speed, power draw and how busy it is.
function gpu_state()
    fields = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw,utilization.gpu"
    temperature, clock, max_clock, power, busy = strip.(split(nvidia_smi(fields), ","))
    return "$(temperature) °C, clock $(clock) (maximum $(max_clock)), $(power), " *
           "$(busy) busy"
end

# The processor's load: the average number of busy cores over the last minute, which
# shows whether other programs shared the machine during the run. The benchmark itself
# keeps one core busy, and all of them while it times the FFTs split over all threads.
function cpu_state()
    load = round(Sys.loadavg()[1]; digits = 1)
    return "$(load) cores busy (last-minute average, including this run)"
end

# The computer and software the run used, as (description => value) lines.
function describe_machine(label)
    machine = [
        "Label" => label,
        "Date" => format(now(), "yyyy-mm-dd HH:MM"),
        "Host" => gethostname(),
        "Processor" => "$(Sys.cpu_info()[1].model), $(Sys.CPU_THREADS) hardware threads",
        "Operating system" => "$(Sys.KERNEL) $(Sys.MACHINE)",
        "Julia" => "$(VERSION), $(Threads.nthreads()) thread(s)",
        "ForceCouplingMethod" => string(pkgversion(ForceCouplingMethod)),
        "BenchmarkTools" => string(pkgversion(BenchmarkTools)),
        "FFTW" => "$(pkgversion(FFTW)) ($(FFTW.get_provider()) library)",
    ]
    if GPU_SKIP_REASON === nothing
        push!(machine, "GPU" => query(() -> CUDA.name(CUDA.device())))
        push!(machine, "GPU identifier" => query(visible_gpu_id))
        push!(machine, "GPU driver" => query(() -> nvidia_smi("driver_version")))
        push!(
            machine,
            "CUDA" => "CUDA.jl $(pkgversion(CUDA)), runtime " *
                      "$(query(CUDA.runtime_version)), driver supports " *
                      "$(query(CUDA.driver_version))",
        )
    end
    return machine
end

# Whether `failure` means the GPU ran out of memory: CUDA.jl reports this with its own
# error type, the FFT library as a failed allocation.
function out_of_gpu_memory(failure)
    failure isa CUDA.OutOfGPUMemoryError && return true
    message = lowercase(sprint(showerror, failure))
    return occursin("out of memory", message) || occursin("alloc_failed", message)
end

# Frees the memory of measurements that are no longer used.
function free_memory(backend)
    GC.gc()
    backend == "gpu" && CUDA.reclaim()
    return nothing
end

# Stores the timings of `group` (one per item) in `results` as
# backend / item / precision / problem.
function store!(results, group, backend, precision, problem)
    for (item, trial) in group
        results[backend][item][precision][problem.name] = trial
    end
    return results
end

# The machine's state now: the GPU's for GPU timings, the processor's otherwise.
state_now(backend) = query(backend == "gpu" ? gpu_state : cpu_state)

# Sets `problem` up in precision `T` on `backend`, warms it up, times the calls that a
# simulation repeats, and stores their timings in `results`. Records the machine state
# after the warm-up in `machine`. Returns the setup measurements, or `nothing` if the GPU
# has no room for the problem.
#
# This is a function of its own so that the problem's data can be freed before setup is
# timed: Julia can keep data alive until the function that used it returns, even after
# its variable is cleared.
function time_repeated_calls!(
    results, machine, backend, ::Type{T}, problem, heading,
) where {T}
    measurements = try
        backend == "gpu" ? gpu_measurements(T, problem) : cpu_measurements(T, problem)
    catch failure
        (backend == "gpu" && out_of_gpu_memory(failure)) || rethrow()
        println("Not enough free GPU memory to set this problem up; skipping it.")
        push!(machine, "Skipped" => "$(heading): not enough free GPU memory to set it up")
        return nothing
    end
    warmup_calls = warm_up!(measurements.case)
    push!(
        machine,
        "State of $(heading) / start" => "after $(warmup_calls) warm-up calls, " *
                                          state_now(backend),
    )
    store!(results, time_each(measurements.calls), backend, string(T), problem)
    return measurements.setup
end

# Times `problem` in precision `T` on `backend` and stores the timings in `results`: the
# repeated calls first, then, once the problem's data is freed, setup (so that the
# largest problems never need two copies in memory at once). Records the machine state
# before and after in `machine`, and notes any part skipped because the GPU ran out of
# memory.
function time_problem!(results, machine, backend, ::Type{T}, problem) where {T}
    heading = "$(uppercase(backend)) / $(problem.name) % / $(T)"
    println("\nTiming ", heading, " (", describe_problem(problem), ")")
    setup = time_repeated_calls!(results, machine, backend, T, problem, heading)
    free_memory(backend)
    setup === nothing && return nothing
    try
        store!(results, run(setup; verbose = true), backend, string(T), problem)
    catch failure
        (backend == "gpu" && out_of_gpu_memory(failure)) || rethrow()
        println("Not enough free GPU memory to time setup; skipping that measurement.")
        push!(machine, "Skipped" => "$(heading): setup timing (not enough free GPU memory)")
        free_memory(backend)
    end
    push!(machine, "State of $(heading) / end" => state_now(backend))
    return nothing
end

label = isempty(ARGS) ? format(now(), "yyyy-mm-dd_HHMMSS") : ARGS[1]
folder = joinpath(@__DIR__, "results", label)
isdir(folder) && error(
    "$(folder) already exists; choose another label or delete that folder first",
)
mkpath(folder)

GPU_SKIP_REASON === nothing || println("Skipping the GPU part: ", GPU_SKIP_REASON, ".")
machine = describe_machine(label)
results = BenchmarkGroup()
for backend in ("cpu", "gpu")
    backend == "gpu" && GPU_SKIP_REASON !== nothing && continue
    for problem in selected_problems(backend), T in (Float32, Float64)
        time_problem!(results, machine, backend, T, problem)
    end
end

BenchmarkTools.save(joinpath(folder, "timings.json"), results)
write_machine(joinpath(folder, "machine.txt"), machine)
write_timings_csv(joinpath(folder, "timings.csv"), results, label)
open(io -> write_report(io, results, machine), joinpath(folder, "report.md"), "w")

write_report(stdout, results, machine)
println("\nResults written to ", folder)
