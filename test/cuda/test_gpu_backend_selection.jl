using Test

# Backend-selection checks that need no GPU. The construction and parity tests
# that require a functional device live in test_gpu_construction.jl, included
# under a `CUDA.functional()` guard.

# The exit code of a new Julia process running `code`. The test session itself loads CUDA
# (for the functional() guard), which provides the real device methods and would mask the
# paths these tests check, so each check runs in a process that loads ForceCouplingMethod
# but not CUDA.
function _exit_code_without_cuda(code)
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --project=$(project) --startup-file=no -e $code`
    return run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull)).exitcode
end

@testset "GPU backend selection without a device" begin
    # `gpu_acceleration = true` must throw a clear `ArgumentError` when the CUDA
    # extension has not been loaded.
    @testset "errors clearly when the CUDA extension is not loaded" begin
        code = """
        using ForceCouplingMethod
        L = (4.0, 6.0, 8.0)
        try
            FFCMConfig(;
                L = L, R_c = 1.0, N = 8, kernel_widths_ratio = 2.0,
                num_grid_points = (Int32(8), Int32(12), Int32(16)), M_G = Int32(6),
                viscosity = 1.0, fft_planning = :estimate, gpu_acceleration = true,
            )
        catch err
            err isa ArgumentError && exit(42)
            rethrow()
        end
        exit(0)
        """
        # exit 42 == threw ArgumentError; 0 == no throw; anything else == crash.
        @test _exit_code_without_cuda(code) == 42
    end

    @testset "calibrating errors clearly when the CUDA extension is not loaded" begin
        code = """
        using ForceCouplingMethod
        try
            ForceCouplingMethod.calibrate(
                [8.0; 8.0; 8.0;;]; L = (16.0, 16.0, 16.0), gpu_acceleration = true,
            )
        catch err
            err isa ArgumentError && occursin("CUDA", err.msg) && exit(42)
            rethrow()
        end
        exit(0)
        """
        @test _exit_code_without_cuda(code) == 42
    end

    @testset "Float64 on the GPU warns once, however many configurations are built" begin
        # A calibration builds many configurations, some in double precision whatever the
        # caller's precision. Without a device, each attempt stops after the warning.
        logs, _ = Test.collect_test_logs() do
            for _ in 1:2
                try
                    _gpu_config(Float64; N = 8)
                catch
                end
            end
        end
        @test count(log -> occursin("Float64", string(log.message)), logs) == 1
    end

    # Stands in for the extension, building on the processor and recording what is asked of
    # it, so that this checks where a calibration sends its calculations on any machine.
    @testset "Each calculation of a calibration goes to the GPU, once memory is freed" begin
        code = """
        using ForceCouplingMethod
        using ForceCouplingMethod:
            CalibrationProblem,
            _assemble_cpu_buffers,
            _candidate_settings,
            _measure_grid,
            _measured_error,
            _random_forces,
            _smallest_accurate_ratios

        const FREED = Ref(0)
        const BUILT = DataType[]
        function ForceCouplingMethod._reclaim_gpu_memory()
            FREED[] += 1
            return 10^10
        end
        function ForceCouplingMethod._assemble_gpu_buffers(
            ::Type{T},
            N::Integer,
            num_grid_points::NTuple{3, Int32},
            M_G::Int32,
            num_cells_total::Integer,
            k_x::Vector{T},
            k_y::Vector{T},
            k_z::Vector{T},
            ::Vector{Int32},
        ) where {T}
            push!(BUILT, T)
            FREED[] ≥ length(BUILT) || exit(1)
            return _assemble_cpu_buffers(
                T, N, num_grid_points, M_G, num_cells_total, k_x, k_y, k_z, :estimate, 1,
            )
        end
        function start_counting()
            empty!(BUILT)
            FREED[] = 0
            return nothing
        end

        # A lattice of 27 spheres, in single precision.
        L = (24.0f0, 24.0f0, 24.0f0)
        Y = zeros(Float32, 3, 27)
        for n in 0:26
            Y[:, n + 1] .= (4 + 8 * mod(n, 3), 4 + 8 * mod(n ÷ 3, 3), 4 + 8 * (n ÷ 9))
        end
        F = _random_forces(Float32, 27)

        problem = CalibrationProblem(
            Y, F, L, 1e-2;
            fft_planning = :estimate, fft_threads = 1, gpu_acceleration = true,
        )
        problem.reference_on_gpu || exit(2)
        BUILT == [Float64, Float64] || exit(3)

        start_counting()
        _measured_error(problem)
        BUILT == [Float32] || exit(4)
        FREED[] == 1 || exit(5)

        start_counting()
        ratios = _smallest_accurate_ratios(problem, 1e-2)
        (length(BUILT) ≥ 3 && all(==(Float32), BUILT)) || exit(6)
        FREED[] == length(BUILT) || exit(7)

        start_counting()
        settings = last(_candidate_settings(problem, ratios))
        seconds = @elapsed _measure_grid(problem, settings, 1.0; lead_in_seconds = 0.01)
        BUILT == [Float32] || exit(8)
        FREED[] == 1 || exit(9)

        # A GPU needs longer than a processor to settle into each grid.
        seconds = @elapsed _measure_grid(problem, settings, 1.0)
        seconds ≥ 1 || exit(10)
        exit(42)
        """
        @test _exit_code_without_cuda(code) == 42
    end
end
