# What the benchmark times. For one problem (problems.jl) in one precision on one backend,
# `cpu_measurements` and `gpu_measurements` set the problem up and return BenchmarkTools
# groups of measurements; run.jl warms each problem up, times it, and frees it before the
# next, since the largest problems need gigabytes of memory each.
#
# Every measurement of a call that a simulation repeats is exactly 50 timed calls, one
# call per timing, made right after one untimed second of the same call. Setup
# measurements take 10 timed calls, since each takes up to a few seconds and a simulation
# sets up only once.

using BenchmarkTools
using CUDA: CUDA, CuArray
using FFTW: FFTW, forget_wisdom
using Logging: Logging
using Random: Xoshiro
using ForceCouplingMethod
using ForceCouplingMethod:
    wrap_positions!,
    assign_cells!,
    sort_particles_by_cell!,
    spread_forces!,
    stokes_solve!,
    interpolate_velocities!,
    correct_velocities!

include(joinpath(@__DIR__, "problems.jl"))

const TIMED_CALLS = 50
const TIMED_SETUP_CALLS = 10
const WARMUP_CALLS = 50
const WARMUP_SECONDS = 10
const MEASUREMENT_WARMUP_SECONDS = 1.0

# The problems each backend runs: comma-separated names (volume fractions in per cent) in
# FFCM_BENCHMARK_CPU_PROBLEMS and FFCM_BENCHMARK_GPU_PROBLEMS, where an empty value skips
# that backend. The GPU runs all five problems by default. The processor runs the two most
# dilute: the denser problems have 4 to 256 times as many particles and are too slow on a
# processor for 50 timed calls of every measurement.
function selected_problems(backend)
    variable = "FFCM_BENCHMARK_$(uppercase(backend))_PROBLEMS"
    default = backend == "cpu" ? "0.05,0.2" : join(PROBLEM_NAMES, ",")
    names = filter(!isempty, strip.(split(get(ENV, variable, default), ",")))
    return [benchmark_problem(name) for name in names]
end

# Why the GPU part cannot run, or `nothing` if it can. Exactly one GPU must be visible:
# on a shared machine the device is chosen explicitly (CUDA_VISIBLE_DEVICES), never by
# default.
function gpu_skip_reason()
    isempty(selected_problems("gpu")) && return "no GPU problems were selected"
    CUDA.functional() || return "no working CUDA GPU is visible"
    CUDA.ndevices() == 1 || return "$(CUDA.ndevices()) GPUs are visible; choose one " *
                                   "with CUDA_VISIBLE_DEVICES=<index>"
    return nothing
end

# A problem ready to time: its configuration, particles and velocity output. One call
# fills the configuration's buffers with a valid state; each pipeline step rewrites its
# outputs from its inputs, so a single step can then be timed over and over on that state.
function ready_problem(::Type{T}, problem, Y, F; config_keywords...) where {T}
    config = FFCMConfig{T}(; problem_keywords(T, problem)..., config_keywords...)
    V = similar(Y)
    mobility!(V, config, Y, F)
    return (; config, Y, F, V)
end

# Runs `mobility!` on `case`, untimed, at least WARMUP_CALLS times and for at least
# WARMUP_SECONDS, and returns the number of calls made. The first calls of a session are
# slow while Julia compiles code and memory settles, and processors and GPUs change their
# clock speed with their load and temperature. A GPU given 50 calls of a small problem (a
# fraction of a second) is still changing speed when the timings start; ten seconds of
# continuous calls bring it to the steady state a long simulation runs in.
function warm_up!(case)
    calls = 0
    start = time()
    while calls < WARMUP_CALLS || time() - start < WARMUP_SECONDS
        mobility!(case.V, case.config, case.Y, case.F)
        calls += 1
    end
    return calls
end

# Sets `benchmark` to exactly `calls` timings of one call each. The time limit is set far
# beyond any run (a million seconds, about eleven days; BenchmarkTools cannot take an
# infinite one), so the count, not a clock, ends the measurement. BenchmarkTools makes one
# more, untimed call first. By default BenchmarkTools also cleans up memory before each
# measurement; that is switched off, because it leaves the GPU idle long enough for its
# clock to drop, and the repeated calls timed here allocate little or no memory.
# `collect_garbage` instead frees memory between timings (outside them), for setup
# measurements, which allocate large buffers.
function timed(benchmark; calls = TIMED_CALLS, collect_garbage = false)
    benchmark.params.samples = calls
    benchmark.params.evals = 1
    benchmark.params.seconds = 1e6
    benchmark.params.gctrial = false
    benchmark.params.gcsample = collect_garbage
    return benchmark
end

# Times each measurement in `group` and returns the timings. Just before its timed calls,
# each measurement runs its own call, untimed, for MEASUREMENT_WARMUP_SECONDS (capped at a
# million calls, far more than any call here fits in a second). Switching from one kind of
# call to another changes how hard the processor and GPU work, and they take a moment to
# adjust their speed; on calls of a millisecond or two, that adjustment otherwise shows
# in the first timed calls.
function time_each(group)
    timings = BenchmarkGroup()
    for (number, (item, benchmark)) in enumerate(group)
        println("($(number)/$(length(group))) timing ", item)
        run(benchmark; samples = 10^6, seconds = MEASUREMENT_WARMUP_SECONDS)
        timings[item] = run(benchmark)
    end
    return timings
end

# A stand-in for `CUDA.synchronize` on the processor, where every call has finished when
# it returns.
no_synchronization() = nothing

# The seven steps of the method (paper §4), each timed on its own. `Y`, `F` and `V` are
# the positions, forces and velocities the steps read and write: the caller's arrays on
# the processor, the configuration's copies on the GPU. `synchronize` waits until the GPU
# has finished, so that a GPU timing covers the work and not only its launch.
function add_steps!(group, config, Y, F, V, synchronize)
    Y_wrapped = config.particles.Y_wrapped
    group["wrap_positions"] = timed(
        @benchmarkable (wrap_positions!($Y_wrapped, $Y, $(config.L)); $synchronize())
    )
    group["assign_cells"] = timed(
        @benchmarkable (assign_cells!($config, $Y_wrapped); $synchronize())
    )
    group["sort_particles_by_cell"] = timed(
        @benchmarkable (sort_particles_by_cell!($config, $Y_wrapped, $F); $synchronize())
    )
    group["spread_forces"] = timed(
        @benchmarkable (spread_forces!($config); $synchronize())
    )
    group["stokes_solve"] = timed(
        @benchmarkable (stokes_solve!($config); $synchronize())
    )
    group["interpolate_velocities"] = timed(
        @benchmarkable (interpolate_velocities!($V, $config); $synchronize())
    )
    group["correct_velocities"] = timed(
        @benchmarkable (correct_velocities!($V, $config); $synchronize())
    )
    return group
end

# The processor measurements of `problem` in precision `T`: the ready problem (for the
# warm-up), the measurements of repeated calls, and the setup measurements.
function cpu_measurements(::Type{T}, problem) where {T}
    Y, F = paper_suspension(T, problem)
    case = ready_problem(T, problem, Y, F)
    config, V = case.config, case.V

    calls = BenchmarkGroup()
    calls["mobility"] = timed(@benchmarkable mobility!($V, $config, $Y, $F))
    add_steps!(calls, config, Y, F, V, no_synchronization)

    # The Stokes solve with quick FFT plans (picked without searching) against searched
    # plans, and with the FFT split across all Julia threads. FFTW's saved plans are
    # forgotten first: FFTW reuses them even when asked for quick plans, which would hide
    # the difference.
    forget_wisdom()
    quick = ready_problem(T, problem, Y, F; fft_planning = :estimate)
    searched = ready_problem(T, problem, Y, F; fft_planning = :measure)
    calls["stokes_solve_quick_plans"] = timed(
        @benchmarkable stokes_solve!($(quick.config))
    )
    calls["stokes_solve_searched_plans"] = timed(
        @benchmarkable stokes_solve!($(searched.config))
    )
    if Threads.nthreads() > 1
        threaded = ready_problem(T, problem, Y, F; fft_threads = Threads.nthreads())
        calls["stokes_solve_threaded_plans"] = timed(
            @benchmarkable stokes_solve!($(threaded.config))
        )
    end

    # Setup with and without FFTW's saved plans. At setup FFTW searches for the fastest way
    # to transform this grid (`fft_planning = :measure`, the default) and remembers the
    # answer for the rest of the session, so a repeated setup skips the search. Forgetting
    # the saved plans before each timing measures a first setup.
    keywords = problem_keywords(T, problem)
    setup = BenchmarkGroup()
    setup["setup_without_saved_plans"] = timed(
        @benchmarkable(FFCMConfig{$T}(; $keywords...), setup = (forget_wisdom()));
        calls = TIMED_SETUP_CALLS, collect_garbage = true,
    )
    setup["setup_with_saved_plans"] = timed(
        @benchmarkable(FFCMConfig{$T}(; $keywords...));
        calls = TIMED_SETUP_CALLS, collect_garbage = true,
    )
    return (; case, calls, setup)
end

# The GPU measurements of `problem` in precision `T`, as for the processor.
function gpu_measurements(::Type{T}, problem) where {T}
    Y, F = paper_suspension(T, problem)
    case = ready_problem(T, problem, Y, F; gpu_acceleration = true)
    config, V = case.config, case.V
    on_gpu = config.particles  # the configuration's GPU copies of Y, F and V

    calls = BenchmarkGroup()
    # The whole call as users see it: `mobility!` copies the positions and forces from the
    # computer's memory to the GPU and the velocities back. Copying the velocities back
    # waits for the GPU to finish, so no extra synchronisation is needed.
    calls["mobility"] = timed(@benchmarkable mobility!($V, $config, $Y, $F))

    # The whole call with the data already on the GPU: the GPU's own share of the work.
    Y_gpu, F_gpu, V_gpu = CuArray(Y), CuArray(F), CuArray(V)
    calls["mobility_gpu_data"] = timed(
        @benchmarkable (mobility!($V_gpu, $config, $Y_gpu, $F_gpu); CUDA.synchronize())
    )

    # The copies alone: positions and forces to the GPU, velocities back.
    calls["copies"] = timed(
        @benchmarkable begin
            copyto!($(on_gpu.Y_input), $Y)
            copyto!($(on_gpu.F_input), $F)
            copyto!($V, $(on_gpu.V_output))
        end
    )

    add_steps!(
        calls, config, on_gpu.Y_input, on_gpu.F_input, on_gpu.V_output, CUDA.synchronize,
    )

    # Setup from scratch: GPU memory and FFT plans. Before each timing (outside it) the
    # previous setup's memory and FFT plans are handed back to the GPU, so every timing
    # reserves its memory and makes its FFT plans afresh, as a simulation's first setup
    # does. This also leaves room for the FFT library's own working memory, which it
    # reserves outside CUDA.jl's memory pool. The double-precision warning that every GPU
    # setup prints is silenced while timing.
    keywords = (; problem_keywords(T, problem)..., gpu_acceleration = true)
    setup = BenchmarkGroup()
    setup["setup"] = timed(
        @benchmarkable(
            FFCMConfig{$T}(; $keywords...),
            setup = (GC.gc(); CUDA.reclaim(); Logging.disable_logging(Logging.Warn)),
            teardown = (Logging.disable_logging(Logging.BelowMinLevel)),
        );
        calls = TIMED_SETUP_CALLS, collect_garbage = true,
    )
    return (; case, calls, setup)
end

const GPU_SKIP_REASON = gpu_skip_reason()
