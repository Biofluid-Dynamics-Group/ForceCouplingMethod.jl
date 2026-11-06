# The mobility operator

ForceCouplingMethod.jl evaluates the mobility operator ``\mathcal{M}^{\mathcal{V}\mathcal{F}}`` of
Su & Keaveny (2024): the linear map taking the forces
``\mathcal{F} = (\boldsymbol{F}_n)_{n=1}^N`` on ``N`` unit-radius spheres at positions
``\mathcal{Y} = (\boldsymbol{Y}_n)_{n=1}^N`` to the velocities
``\mathcal{V} = (\boldsymbol{V}_n)_{n=1}^N`` they induce through the triply-periodic
Stokes flow (§2, equations (5)–(6)). In the force-coupling method each point force is
regularised by a Gaussian kernel whose width ``\sigma = a/\sqrt{\pi}`` is set by the
particle radius ``a``, chosen so that a single sphere recovers the Stokes drag law (§2).

## The fast-FCM splitting

Resolving the width-``\sigma`` kernel directly ties the grid resolution of the Stokes
solver to the particle radius. Fast FCM decouples the two through the splitting

```math
\mathcal{M}^{\mathcal{V}\mathcal{F}}
= \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}
+ \left(\mathcal{M}^{\mathcal{V}\mathcal{F}} - \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}\right)
\qquad \text{(§3, equation (19))}.
```

The smooth part ``\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}`` is the FCM mobility of a
*wider* kernel, of width ``\Sigma \geq \sigma``, which the spectral steps — spreading,
the Stokes solve, and interpolation — resolve on a correspondingly coarser grid. The
remainder ``\mathcal{M}^{\mathcal{V}\mathcal{F}} - \tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}``
decays like a Gaussian in the pair separation, so it is applied as a sparse real-space
pairwise sum over neighbours within a cutoff ``R_c``. The corrected result recovers the
``\sigma``-regularised mobility *independent of* ``\Sigma``: the ratio ``\Sigma/\sigma``
and the cutoff ``R_c`` become tuning parameters that trade grid work against pair work
(§5), which [Choosing parameters](../choosing-parameters.md) explains how to set.

## The six-step pipeline

[`mobility!`](@ref)`(V, config, Y, F)` composes the six sub-algorithms of §4, applied in
order. Step 1 (spatial hashing) is two calls — the position wrap and the cell hash — so
the six steps are seven calls:

```julia
wrap_positions!(config.particles.Y_wrapped, Y, config.L)        # step 1: fold into [0, L)
assign_cells!(config, config.particles.Y_wrapped)               # step 1: cell hash
sort_particles_by_cell!(config, config.particles.Y_wrapped, F)  # step 2: cell list + gather
spread_forces!(config)                                # step 3: J̃†  (forces → density)
stokes_solve!(config)                                 # step 4: L⁻¹ (FFT Stokes solve)
interpolate_velocities!(V, config)                    # step 5: J̃   (velocity → particles)
correct_velocities!(V, config)                        # step 6: + (M − M̃) real-space
```

Steps 1–2 build the [cell list](cell-lists.md) and gather the forces into cell order;
steps 3–5 compute ``\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}\mathcal{F}`` (the
[spread](spreading-interpolation.md), the [Stokes solve](stokes-solve.md), and the
[interpolation](spreading-interpolation.md)); step 6 adds the real-space remainder (the
[pairwise correction](pairwise-correction.md)). Interpolation and the correction both
write `V` in the caller's original particle order — interpolation scatters back through
the step-2 permutation, and the correction adds onto the same buffer — so the composition
needs no final reordering.

## Symmetric positive-definiteness

The assembled mobility is symmetric positive-definite — the property a
conjugate-gradient resistance solve depends on (§3.3, "Positive splitting"). Writing
``S`` for the spread matrix (entry ``S_{g,n} = \tilde{\Delta}_n(\boldsymbol{x}_g; \Sigma)``,
the weight particle ``n`` contributes to grid point ``g``), ``P`` for the step-2 sort
permutation, and ``\mathcal{L}^{-1}`` for the Stokes solve, the smooth part assembles as

```math
\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}} = h^3 (SP)^T \mathcal{L}^{-1} (SP).
```

``\mathcal{L}^{-1}`` is symmetric positive-semidefinite: per Fourier mode it is the
orthogonal projector ``\boldsymbol{I} - \hat{\boldsymbol{k}}\hat{\boldsymbol{k}}^T``
scaled by the positive factor ``1/(\mu k^2)``. Interpolation being the *exact* discrete
adjoint of spreading — the uniform ``h^3`` quadrature weight, see
[why the uniform h³ weight](@ref "Why the uniform h³ weight") —
makes the sandwich symmetric positive-semidefinite for any ``S``, and positive-definite
for distinct-particle configurations. The real-space correction is a symmetric pair
tensor that vanishes at ``\Sigma = \sigma`` (see the
[pairwise correction](pairwise-correction.md)), so the splitting preserves the
structure.
 The [`FFCMMobility`](@ref) operator declares `issymmetric` and `isposdef`
accordingly.

## The matrix-free interface

Iterative solvers see flat vectors; the pipeline works on ``3 \times N`` matrices. The
convention bridging them is the column-major `vec` of a ``3 \times N`` matrix: entry
``3(n-1) + i`` holds axis ``i \in \{1, 2, 3\}`` of particle ``n``, so a flat velocity
vector is equivalently `reshape(v, 3, N)`.

[`FFCMMobility`](@ref)`(config, Y)` closes over a configuration and a fixed position
matrix and implements the `LinearAlgebra` operator interface: `size(M) == (3N, 3N)`,
allocation-free three- and five-argument `mul!`, and the allocating convenience `M * F`
in the natural ``3 \times N`` layout. The operator drops into `IterativeSolvers.gmres!`,
`KrylovKit.linsolve`, or any `mul!`-based solver without a call-site adapter.
