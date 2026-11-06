module ForceCouplingMethod

using StaticArrays: SVector
using StructArrays: StructArray, components
using FFTW: ESTIMATE, MEASURE, PATIENT, plan_rfft, plan_brfft
using LinearAlgebra: dot
using Random: Xoshiro
using SpecialFunctions: erf

import LinearAlgebra: mul!, issymmetric, isposdef

include("config.jl")
include("cell_list.jl")
include("stencil.jl")
include("spread_forces.jl")
include("stokes_solve.jl")
include("interpolate.jl")
include("correct_velocities.jl")
include("mobility.jl")
include("calibrate.jl")

export FFCMConfig, mobility!, FFCMMobility
public calibrate

end