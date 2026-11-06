# Notation

A quick reference mapping the symbols of Su & Keaveny (2024) to the code identifiers
used throughout this documentation. Each method page introduces only the symbols new to
its step; the symbols shared across steps are collected here.

!!! note
    Paper section and equation numbers cited throughout this documentation have been
    verified against the published article (Su & Keaveny 2024, *J. Comput. Phys.* 510,
    113060). The one exception is the ``\operatorname{erf}``-argument typo in its
    equation (31), documented on the [pairwise correction](method/pairwise-correction.md)
    page.

## Index conventions

- ``i \in \{x, y, z\} \equiv \{1, 2, 3\}`` is a Cartesian axis index.
- ``n, m \in \{1, \dots, N\}`` are particle indices.
- Particle data live in column-major ``3 \times N`` matrices, so `Y[i, n]` is component
  ``i`` of particle ``n`` (the code indexes from 1).
- ``(i_x, i_y, i_z)`` is a 1-based grid-point index on the spectral grid: the grid point
  is ``\boldsymbol{x}_g = \bigl((i_x - 1)h, (i_y - 1)h, (i_z - 1)h\bigr)``. The same
  letters index the Fourier grid, where they address the wavenumbers
  `k_x[i_x]`, `k_y[i_y]`, `k_z[i_z]`.

## Symbols

- `L` ``= (L_x, L_y, L_z)`` — the lengths of the periodic box, so the domain is
  ``\Omega = [0, L_x) \times [0, L_y) \times [0, L_z)``.
- `N` ``= N`` — the number of particles.
- `Y` ``= (\boldsymbol{Y}_n)_{n=1}^N`` — the particle positions; ``\boldsymbol{Y}_n`` is
  the position of particle ``n``.
- `F` ``= (\boldsymbol{F}_n)_{n=1}^N`` — the forces on the particles.
- `V` ``= (\boldsymbol{V}_n)_{n=1}^N`` — the particle velocities the operator returns.
- `R_c` ``= R_c`` — the cutoff radius of the pairwise correction.
- `num_cells` ``= (m_x, m_y, m_z)`` — the number of cells per axis in the cell list.
- `cell_size` ``= (L_x/m_x, L_y/m_y, L_z/m_z)`` — the cell extents per axis;
  `inv_cell_size` holds their reciprocals (precomputed for the hot path).
- `cell_hash` — `cell_hash[n]` is the linear cell index of particle ``n``.
- `original_index` — the sort permutation, sorted slot ``\mapsto`` original particle
  index (``n = \text{original index}[s]``).
- `cell_start`, `cell_end` — the per-cell 1-based **inclusive** range of sorted slots;
  `counting_sort_scratch` is the counting-sort scratch (the per-cell histogram, then the
  per-cell write cursor).
- `a` ``= a`` — the particle radius; ``a = 1`` in the current implementation.
- `σ` ``= \sigma`` — the FCM Gaussian width, ``\sigma = a/\sqrt{\pi}``.
- `Σ` ``= \Sigma`` — the modified (fast-FCM) kernel width, with ``\Sigma \geq \sigma``.
- `Σ_over_σ` ``= \Sigma/\sigma`` — the resolution control parameter (the
  `kernel_widths_ratio` keyword).
- `μ` ``= \mu`` — the dynamic viscosity. The paper writes ``\eta`` for this quantity.
- `num_grid_points` ``= (M_x, M_y, M_z)`` — the grid points per axis of the spectral
  grid; the total is ``M = M_x M_y M_z``.
- `M_G` ``= M_G`` — the per-axis width of the cubic ``M_G \times M_G \times M_G`` kernel
  stencil.
- `h` ``= h`` — the uniform grid spacing ``h = L_i/M_i``, equal across axes. §5 of the
  paper, and the [Choosing parameters](choosing-parameters.md) page, write ``\Delta x``.
- `kernel_resolution` ``= \Sigma/\Delta x`` — the grid points per Gaussian width.
- `cutoff_ratio` ``= \lambda_\varepsilon = R_c/\Sigma`` — the reach of the pairwise
  correction in Gaussian widths (§5.2).
- `relative_tolerance` ``= \varepsilon`` — the largest error a calibration accepts, in the
  measure of §5.1, equation (69).
- ``\phi = \tfrac{4}{3}\pi a^3 N/(L_x L_y L_z)`` — the volume fraction of the suspension.
- `k_x`, `k_y`, `k_z` — the per-axis wavenumbers ``k_i = 2\pi n_i/L_i`` of the spectral
  grid, with ``n_i`` the signed Fourier index; together they form the wavenumber vector
  ``\boldsymbol{k}``.
- `force_density` ``= \boldsymbol{f}`` and `fluid_velocity` ``= \boldsymbol{u}`` — the
  spread force density and the fluid velocity, sampled on the spectral grid.
- `fluid_hat` — the Fourier transform of `force_density` on entry to the Stokes solve,
  overwritten with that of `fluid_velocity` on exit.

## Operators and kernels

- ``\mathcal{M}^{\mathcal{V}\mathcal{F}}`` — the mobility operator, mapping the forces
  ``\mathcal{F}`` at positions ``\mathcal{Y}`` to the velocities ``\mathcal{V}`` they
  induce through the periodic Stokes flow.
- ``\tilde{\mathcal{M}}^{\mathcal{V}\mathcal{F}}`` — the modified mobility, built from
  the width-``\Sigma`` kernel; the spectral steps compute this and the pairwise
  correction recovers ``\mathcal{M}^{\mathcal{V}\mathcal{F}}``.
- ``\tilde{\mathcal{J}}^\dagger`` and ``\tilde{\mathcal{J}}`` — the spreading operator
  and its adjoint, interpolation.
- ``\mathcal{L}^{-1}`` — the inverse periodic Stokes operator.
- ``\Delta_n(\boldsymbol{x}; w)`` — an isotropic Gaussian of width ``w`` centred at
  ``\boldsymbol{Y}_n``.
- ``\tilde{\Delta}_n(\boldsymbol{x}; \Sigma)`` — the modified (fast-FCM) kernel.
- ``\Delta`` — the Laplacian. The paper writes ``\nabla^2``.
- ``\boldsymbol{G}, \boldsymbol{S}, \boldsymbol{Q}, \boldsymbol{T}`` — the Stokeslet
  (Oseen tensor) and the operators of the FCM mobility expansion entering the real-space
  correction; their closed forms are on the
  [pairwise correction](method/pairwise-correction.md) page.
- ``A, B, \delta`` — the two pair scalars of the correction tensor
  ``A\boldsymbol{I} + B\,\boldsymbol{x} \otimes \boldsymbol{x}`` and the self-correction
  scalar; code `isotropic_coefficient`, `parallel_coefficient`, `self_correction_term`.
