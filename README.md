# ForceCouplingMethod.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://biofluid-dynamics-group.github.io/ForceCouplingMethod.jl/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://biofluid-dynamics-group.github.io/ForceCouplingMethod.jl/dev/)
[![CI](https://github.com/Biofluid-Dynamics-Group/ForceCouplingMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Biofluid-Dynamics-Group/ForceCouplingMethod.jl/actions/workflows/CI.yml)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

Julia implementation of the Fast Force-Coupling Method (FFCM) ([Su & Keaveny (2024), *Accelerating the
force-coupling method for hydrodynamic interactions in periodic
domains*, J. Comput. Phys. **510**, 113060](https://www.sciencedirect.com/science/article/pii/S0021999124003097)), for
hydrodynamic interactions between rigid spheres in a triply-periodic
Stokes flow.

FFCM is a numerical approximation to the mobility tensor of a collection of identical spherical rigid particles
suspended in a fluid that obeys the Stokes equation with periodic boundary conditions. Given $N$ spherical particles of radius $a=1$ located at $\left(\boldsymbol{Y}_n\right) _{n = 1}^N \subset \Omega$
with forces $\left(\boldsymbol{F}_n\right) _{n = 1}^N$ acting on them, where $\Omega \cong \mathbb{T}^3$, FFCM computes
their resulting velocities $\left(\boldsymbol{V}_n\right) _{n = 1}^N$ considering hydrodynamic interactions.

This package currently _only_ implements the linear force-velocity relationship for each sphere.

## Installation

ForceCouplingMethod.jl is registered in the Julia General registry: from the Julia REPL, enter
package mode with `]` and run

```julia-repl
pkg> add ForceCouplingMethod
```

## Usage

The package provides three objects for plug-and-play usage of the method. `FFCMConfig` provides
the setup for the problem, where the physics (domain size, number of particles, viscosity) and
numerical parameters (spectral method grid, local stencil size, method parameters) can be set.

`FFCMMobility` is the actual mobility operator, and `mobility!` is a mutating function that acts on its first argument, populating it with the
velocities given by the arguments (both interfaces share the same backend).

```julia
using ForceCouplingMethod

T = Float32  # or Float64

L_x = T(10)  # Set \Omega = [0, 10)^3
L_y = T(10)
L_z = T(10)

μ = T(1)  # Viscosity

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
    viscosity = μ,
    num_grid_points = (num_points, num_points, num_points),
    M_G = M_G,
)

V = zeros(T, 3, N)  # Target array for velocities, size 3xN of type T
mobility!(V, config, Y, F)  # Use FFCM as a function

M = FFCMMobility(config, Y)  # Mobility operator
V = M * F  # Alternatively, use the method as a linear operator (useful for iterative solvers)
```

The first mobility call after loading the package will trigger compilation, but subsequent calls should run at full speed.

## Parameter optimisation

The method has four numerical parameters (paper §5):
- $M_G$ (`M_G`), the grid points the Gaussian covers in one direction
- $\Sigma/\Delta x$ (through `num_grid_points`), the grid points per Gaussian width
- $\Sigma/\sigma$ (`kernel_widths_ratio`), how much wider the computational Gaussian is than the particle's
- $R_c/\sigma$ (`R_c`), the reach of the pair correction

Their best values depend on how the particles are arranged and on your machine. `ForceCouplingMethod.calibrate` finds the fastest settings that meet an error you choose, on your own particles and machine, by timing real calls. For the example above:

```julia
L = (L_x, L_y, L_z)
settings = ForceCouplingMethod.calibrate(Y; L, relative_tolerance = 1e-4)
config = FFCMConfig(; L, N, viscosity = μ, settings...)
```

It logs a report in the paper's notation. Calibrating runs on the processor for now; `gpu_acceleration = true` is not yet implemented for it. See [Choosing parameters](https://biofluid-dynamics-group.github.io/ForceCouplingMethod.jl/dev/choosing-parameters/) for what it does, what it costs and its limits.

## GPU acceleration

A GPU parallel implementation that follows the paper authors' reference CUDA
implementation, [cuFCM](https://github.com/racksa/cuFCM), is available as a package
extension dependent on [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

First, select a GPU by setting a single integer as the `CUDA_VISIBLE_DEVICES` environment
variable in your terminal:
```bash
export CUDA_VISIBLE_DEVICES=<index>
```

On Windows, set the same variable in PowerShell (`$env:CUDA_VISIBLE_DEVICES = "<index>"`)
or use WSL2 — the GPU backend does not require a Unix shell.

> On one of Imperial College London's Mathematics Deparment NVIDIA clusters, where this package was tested, you can check on the
> available cards using 
> ```bash
> gpustat
> ```
> and then pin the `<index>` of whichever NVIDIA card is available to `CUDA_VISIBLE_DEVICES`.

If a single card set in `CUDA_VISIBLE_DEVICES` and [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
is available in your environment, loading it alongside ForceCouplingMethod and
adding `gpu_acceleration = true` to the config constructor will trigger the parallel algorithm:

```julia
using CUDA
using ForceCouplingMethod

config = FFCMConfig(
    L = (L_x, L_y, L_z),
    R_c = R_c,
    N = N,
    kernel_widths_ratio = kernel_widths_ratio,
    viscosity = μ,
    num_grid_points = (num_points, num_points, num_points),
    M_G = M_G,
    gpu_acceleration = true,
)
```

Everything else is unchanged: `mobility!` and `FFCMMobility` take and return ordinary host
arrays, and the whole pipeline runs resident on the device (the host-device traffic is
handled internally). `Float32` is the intended GPU precision; `Float64` works but emits a
warning, since consumer NVIDIA cards run double precision at a small fraction of
single-precision throughput. For the same inputs, the GPU backend reproduces the CPU
backend's velocities to round-off.

### Driver compatibility

[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) supports various NVIDIA drivers, but in order for
the code to run, the CUDA.jl version needs to match the driver on the card that you want to use.
CUDA.jl provides the function `CUDA.functional()` that returns `true` if the installation is compatible
with the current hardware and drivers and the code is ready to run. `ForceCouplingMethod` supports CUDA.jl `5.8`-`6.2`, and the
package manager installs the newest by default. On older drivers the newest
series would report `CUDA.functional() == false` and the GPU backend would not be able to be used.

If `using CUDA; CUDA.functional()` returns `false` on your machine, the issue might be a version-driver
mismatch. If updating the drivers isn't a possibility at the moment, installing an older CUDA.jl version and pinning it in your `Manifest.toml` might help. This looks something like
```julia-repl
pkg> add CUDA@5.8.5
pkg> pin CUDA
```
in the Julia REPL package mode. See the
[CUDA.jl documentation](https://cuda.juliagpu.org) for version-driver compatibility.

## LLM assistance

This repository was written with assistance of `claude-5.5-opus`, `claude-5-fable`, `claude-4.8-opus`, `claude-4.7-opus`, `claude-5-sonnet` and `claude-4.6-sonnet` in code generation, test generation, design and documentation.

## Contributing

Contributions are welcome! A general guide for contribution to scientific Julia packages is available by the SciML community as [ColPrac](https://github.com/SciML/ColPrac). We additionally ask to disclose any use of LLM assistance in coding for transparency, and pull requests to be submitted by humans.