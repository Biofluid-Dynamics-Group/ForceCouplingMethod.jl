# ForceCouplingMethod.jl

Julia implementation of the Fast Force-Coupling Method (FFCM) of
[Su & Keaveny (2024), *Accelerating the force-coupling method for hydrodynamic
interactions in periodic domains*, J. Comput. Phys. **510**,
113060](https://doi.org/10.1016/j.jcp.2024.113060), for hydrodynamic
interactions between rigid spheres in a triply-periodic Stokes flow.

FFCM is a numerical approximation to the mobility tensor of a collection of
identical spherical rigid particles suspended in a fluid that obeys the Stokes
equation with periodic boundary conditions. Given ``N`` spherical particles of
radius ``a = 1`` located at ``\boldsymbol{Y}_n`` with forces
``\boldsymbol{F}_n`` acting on them, FFCM computes their resulting velocities
``\boldsymbol{V}_n`` considering hydrodynamic interactions.

## Installation

ForceCouplingMethod.jl is registered in the Julia General registry: from the Julia REPL, enter
package mode with `]` and run

```julia-repl
pkg> add ForceCouplingMethod
```

## Usage

The package provides three objects for plug-and-play usage of the method.
`FFCMConfig` provides the setup for the problem, where the physics (domain
size, number of particles, viscosity) and numerical parameters (spectral
method grid, local stencil size, method parameters) can be set.

`FFCMMobility` is the actual mobility operator, and `mobility!` is a mutating
function that acts on its first argument, populating it with the velocities
given by the arguments (both interfaces share the same backend).

```julia
using ForceCouplingMethod

T = Float32  # or Float64

L_x = T(10)  # Set \Omega = [0, 10)^3
L_y = T(10)
L_z = T(10)

viscosity = T(1)  # Fluid dynamic viscosity

R_c = T(1)  # Cutoff radius
kernel_widths_ratio = T(2)  # FFCM splitting parameter

Y = T[9.3 4.1 -0.4 4.0; 4.0 4.2 4.0 9.9; 4.0 4.0 4.0 4.0]  # Positions 3xN
F = T[0.5 -0.3 0.2 0.1; -0.1 0.4 0.0 0.3; 0.2 0.1 0.7 -0.5]  # Forces 3xN
N = size(Y, 2)  # Number of particles

num_points = Int32(16)  # Grid discretisation
M_G = 8  # Stencil size

config = FFCMConfig(
    L = (L_x, L_y, L_z),
    R_c = R_c,
    N = N,
    kernel_widths_ratio = kernel_widths_ratio,
    viscosity = viscosity,
    num_grid_points = (num_points, num_points, num_points),
    M_G = M_G,
)

V = zeros(T, 3, N)  # Target array for velocities, size 3xN of type T
mobility!(V, config, Y, F)  # Use FFCM as a function

M = FFCMMobility(config, Y)  # Mobility operator
V = M * F  # Alternatively, use the method as a linear operator
```

The working precision is inferred from the element type of the domain lengths
`L`; it can also be set explicitly with `FFCMConfig{T}(...)`.

The first `mobility!` (or `M * F`) call after loading the package incurs
Julia's just-in-time compilation. For timing or long solves, make one warmup
call on the configuration first; subsequent calls run at full speed.

On a machine with an NVIDIA GPU, constructing with `gpu_acceleration = true`
(after `using CUDA`) runs the whole pipeline on the device with the same API —
see [GPU acceleration](gpu.md).

The numerical settings above are only an example. To find the fastest settings
that meet an error you choose, on your own particles and machine, use
`ForceCouplingMethod.calibrate`, in place of writing them by hand:

```julia
L = (L_x, L_y, L_z)
settings = ForceCouplingMethod.calibrate(Y; L, relative_tolerance = 1e-4)
config = FFCMConfig(; L, N, viscosity, settings...)
```

See [Choosing parameters](choosing-parameters.md).

## Reading this documentation

- The [method pages](method/overview.md) explain the algorithm — the fast-FCM
  splitting and each pipeline step — citing Su & Keaveny (2024) by section and
  equation; [Notation](notation.md) maps the paper's symbols to code
  identifiers.
- [Choosing parameters](choosing-parameters.md), [GPU acceleration](gpu.md) and
  [Performance tips](performance.md) cover practical use.
- The [API reference](api.md) documents the public names; the
  [developer documentation](devdocs/architecture.md) documents the internals.
