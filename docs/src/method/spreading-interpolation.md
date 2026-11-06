# Spreading and interpolation

Steps 3 and 5 of the pipeline (Su & Keaveny 2024, §4) are a transpose pair: spreading
scatters the particle forces onto the spectral grid as a smooth force density, and
interpolation gathers the solved fluid velocity back to the particles. Both use the same
modified Gaussian kernel over the same per-particle stencil, and that sharing is not a
convenience but a structural requirement — it is what makes the assembled mobility
symmetric (see below).

## The FCM kernels

The original FCM kernel is the isotropic Gaussian centred on particle ``n``,

```math
\Delta_n(\boldsymbol{x}; \sigma)
= \frac{1}{(2\pi\sigma^2)^{3/2}}\,
  e^{-\lvert \boldsymbol{x} - \boldsymbol{Y}_n \rvert^2 / 2\sigma^2}
\qquad (\text{§2, equation (1)}),
```

with the width ``\sigma = a/\sqrt{\pi}`` fixed by the particle radius so that a single
sphere recovers the Stokes drag law (§2). All lengths in the package are expressed in
units of ``a``, so ``a = 1``. Fast FCM spreads with the *modified* kernel at the wider
width ``\Sigma \geq \sigma``,

```math
\tilde{\Delta}_n(\boldsymbol{x}; \Sigma)
= \left(1 + \frac{\sigma^2 - \Sigma^2}{2} \Delta\right) \Delta_n(\boldsymbol{x}; \Sigma)
\qquad (\text{§3, equation (22)}),
```

where ``\Delta`` denotes the Laplacian. A wider kernel reduces the grid resolution the
Stokes solver needs; the hydrodynamic error this introduces decays like a Gaussian in
the pair separation and is removed by the [pairwise correction](pairwise-correction.md).
Evaluating the Laplacian of the Gaussian gives the closed form the code uses,

```math
\tilde{\Delta}_n(\boldsymbol{x}; \Sigma) = \left(a_0 + a_2 r_n^2\right)
\Delta_n(\boldsymbol{x}; \Sigma),
\qquad
a_0 = 1 - \frac{3(\sigma^2 - \Sigma^2)}{2\Sigma^2},
\quad
a_2 = \frac{\sigma^2 - \Sigma^2}{2\Sigma^4},
```

with ``r_n = \lvert \boldsymbol{x} - \boldsymbol{Y}_n \rvert``. At ``\Sigma = \sigma``
the coefficients collapse to ``a_0 = 1``, ``a_2 = 0`` and the method reduces to standard
FCM — supported because it is free and useful for validation.

## Spreading forces to the grid

The spreading operator maps the ``N`` forces to a smooth force density,

```math
\tilde{\mathcal{J}}^\dagger[\mathcal{F}](\boldsymbol{x})
= \sum_{n=1}^N \boldsymbol{F}_n \tilde{\Delta}_n(\boldsymbol{x}; \Sigma)
\qquad (\text{§3, equation (25)}).
```

Sampled on the grid, each particle contributes only to the ``M_G^3`` grid points of its
stencil — the Gaussian decays so fast that the truncation error is exponentially small
in ``M_G`` (the ``(M_G, \Sigma/h)`` accuracy calibration is §5, Table 1 and Fig. 1(a)).
Because the kernel integrates to one (the Gaussian integrates to one and the Laplacian
term integrates to zero), the spread field conserves the total force and its first
moment up to that truncation.

### Separability

The isotropic Gaussian factors per axis: with
``g(s; \Sigma) = (2\pi\Sigma^2)^{-1/2} e^{-s^2/2\Sigma^2}``,

```math
\Delta_n(\boldsymbol{x}_g; \Sigma)
= g(x_{i_x} - Y_{n,1})\, g(y_{i_y} - Y_{n,2})\, g(z_{i_z} - Y_{n,3}),
\qquad
r_n^2 = \sum_i (x_i - Y_{n,i})^2 .
```

Per particle, the kernel therefore precomputes per-axis one-dimensional Gaussian weights
and squared distances — ``\mathcal{O}(M_G)`` exponentials per axis — and the inner
``M_G^3`` loop assembles ``\tilde{\Delta}_n`` from three multiplies and the polynomial
factor ``(a_0 + a_2 r_n^2)``, evaluating no further exponentials.

### Stencil anchoring

For axis ``i`` and particle ``n`` the stencil is anchored at the nearest grid point,
``j_i = \operatorname{round}(Y_{n,i}/h)`` with ties rounding to even, and covers the
grid indices ``j_i - \lfloor M_G/2 \rfloor + s`` for ``s = 0, \dots, M_G - 1``, wrapped
periodically ``\bmod\ M_i``. For odd ``M_G`` the stencil is symmetric about the anchor;
for even ``M_G`` it covers one more point below than above, which makes
[odd widths the efficient choice](@ref "Odd stencil widths"). Nearest-grid-point anchoring
is the convention the ``(M_G, \Sigma/h)`` calibration of §5 (Table 1) assumes. A
particle near the domain edge spreads to a stencil that wraps to the opposite side,
identically to an interior particle.

## Interpolating velocities from the grid

The interpolation operator is the kernel-weighted volume average of the fluid velocity,

```math
\boldsymbol{V}_n = \tilde{\mathcal{J}}[\boldsymbol{u}]_n
= \int_\Omega \boldsymbol{u}(\boldsymbol{x})\, \tilde{\Delta}_n(\boldsymbol{x}; \Sigma)
\, \mathrm{d}\boldsymbol{x}
\qquad (\text{§3, equation (26)}).
```

With ``\boldsymbol{u}`` known at the grid points, the integral is approximated by the
trapezoidal rule with the uniform weight ``h^3`` over the same ``M_G^3`` stencil as the
spread:

```math
\boldsymbol{V}_n = h^3 \sum_{\text{stencil}}
\boldsymbol{u}(\boldsymbol{x}_g) \left(a_0 + a_2 r_n^2\right) g_x g_y g_z .
```

The only differences from the spread are the direction — a gather from the grid rather
than a scatter to it — and the quadrature factor ``h^3``, applied once per particle.
The result is written in the caller's original particle order (the step-2 sort
permutation is folded into the output write).

### Why the uniform h³ weight

Two independent reasons fix both the quadrature rule and its weight; neither is a free
choice.

**Exact discrete adjointness makes the mobility symmetric.** Let ``S`` be the spread
matrix, ``S_{g,n} = \tilde{\Delta}_n(\boldsymbol{x}_g; \Sigma)``. The trapezoidal
interpolation above is exactly the matrix ``h^3 S^T`` acting on the grid velocity, so
the assembled smooth mobility is the symmetric sandwich
``h^3 (SP)^T \mathcal{L}^{-1} (SP)`` described in the
[overview](@ref "Symmetric positive-definiteness"). This holds *only* because
interpolation is the exact transpose of spreading: a non-uniform quadrature weight would
replace ``h^3`` by a diagonal ``W \neq h^3\boldsymbol{I}``, and Gauss-type nodes would
change the node set entirely — either breaks the transpose relation and with it the
symmetry a conjugate-gradient resistance solver depends on. In the implementation the
spread and the interpolation share one stencil-fill routine, so the interpolation
weights are bit-for-bit the spread weights and the adjointness is exact in floating
point, not just in exact arithmetic.

**The trapezoidal rule is spectrally accurate here.** On a periodic domain the
trapezoidal rule converges faster than any power of ``h`` for smooth integrands. The
Euler–Maclaurin formula writes its error as a series of boundary terms in the odd
derivatives at the interval ends; for an ``L``-periodic integrand every derivative
matches at the endpoints and every term cancels. The integrand here — the product of the
discrete velocity field (a trigonometric polynomial) and the periodised Gaussian kernel
— is ``C^\infty`` and periodic, so the full-grid trapezoidal sum is spectrally accurate,
and restricting it to the ``M_G^3`` stencil drops only the exponentially small Gaussian
tail.
