module ForceCouplingMethodCUDAExt

# CUDA backend of the Fast FCM mobility operator (Su & Keaveny 2024, §3 and §4).
# Loaded automatically when a user runs `using CUDA` alongside `using ForceCouplingMethod`,
# through the package's `[weakdeps]`/`[extensions]` tables. This is the only place
# CUDA appears; `src` stays free of it.
#
# The extension mirrors the `src/` layout, one file per pipeline concern:
# `config.jl` builds the device buffers and cuFFT plans; `cell_list.jl` holds the
# step-1 device kernels (wrap, hash, counting sort, gather).

using ForceCouplingMethod
using CUDA
using LinearAlgebra: dot
using StaticArrays: SVector
using StructArrays: StructArray, components

import ForceCouplingMethod:
    _apply_inverse_stokes_kernel!,
    _assemble_gpu_buffers,
    _assign_cells_kernel!,
    _build_cell_list_kernel!,
    _correct_velocities_kernel!,
    _correction_scalars,
    _gather_particles_kernel!,
    _interpolate_velocities_kernel!,
    _min_image,
    _modified_kernel_coefficients,
    _reclaim_gpu_memory,
    _retrieve_mobility_output!,
    _self_correction,
    _spread_forces_kernel!,
    _stage_mobility_io!,
    wrap_positions!

include("config.jl")
include("cell_list.jl")
include("spread_forces.jl")
include("stokes_solve.jl")
include("interpolate.jl")
include("correct_velocities.jl")
include("mobility.jl")

end # module ForceCouplingMethodCUDAExt
