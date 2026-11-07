# Choosing parameters

The settings of the method decide how accurate it is and how fast it runs, and the best
values depend on how the particles are arranged and on the machine. §5 of Su & Keaveny
(2024) sets out how to find them and encourages users to explore.
[`ForceCouplingMethod.calibrate`](@ref) does that exploring in one call, on your own
particles and machine.

## The four settings

§5 describes the method by four numbers. The first two set how accurately the grid
represents the Gaussian. The last two trade grid work against pair work.

| Setting | Meaning | In [`FFCMConfig`](@ref) |
|---|---|---|
| ``M_G`` | Grid points the Gaussian covers in one direction | `M_G` |
| ``\Sigma/\Delta x`` | Grid points per Gaussian width, with ``\Delta x`` the grid spacing | through `num_grid_points` |
| ``\Sigma/\sigma`` | How much wider the computational Gaussian is than the particle's | `kernel_widths_ratio` |
| ``R_c/\sigma`` | The reach of the pair correction | `R_c` |

For a given accuracy, the cutoff grows in proportion to the Gaussian, along the line
``R_c/\sigma = \lambda_\varepsilon \Sigma/\sigma`` (§5.2, Fig. 1(b)). The slope
``\lambda_\varepsilon = R_c/\Sigma`` is not a further setting, but it is the convenient
number to search for: it barely changes from grid to grid, so once it is known every grid's
``R_c`` follows from it.

Accuracy is measured as in §5.1, equation (69): the error in each particle's velocity
relative to the size of that velocity, averaged over the particles,

```math
\varepsilon = \frac{1}{N} \sum_{n=1}^N
\frac{\lVert \boldsymbol{V}_n - \boldsymbol{V}_n^\star \rVert}{\lVert \boldsymbol{V}_n^\star \rVert},
```

where ``\boldsymbol{V}^\star`` are velocities computed very accurately. The exact answer is
standard FCM, which fast FCM reproduces at unlimited resolution whatever ``\Sigma/\sigma``
is (§3).

## Calibrating

Give `calibrate` the positions, the box and the error you can accept:

```@example calibrate
using ForceCouplingMethod

L = (32.0, 32.0, 32.0)
Y = 32 .* rand(3, 150)  # 150 spheres at random positions

settings = ForceCouplingMethod.calibrate(Y; L, relative_tolerance = 1e-3)
nothing # hide
```

It returns the settings:

```@example calibrate
settings
```

They go straight into the constructor:

```@example calibrate
config = FFCMConfig(; L, N = size(Y, 2), viscosity = 1.0, settings...)
nothing # hide
```

`calibrate` is public but not exported, because it is a tool you call once, by its full
name. Only `Y` and `L` are required, and the [API reference](api.md) lists every keyword.
Four deserve a word:

- `relative_tolerance` is the ``\varepsilon`` you can accept. It defaults to ``10^{-4}``,
  the value §5.2 works with.
- `F` holds the forces to measure the error with. By default they are random and repeatable,
  as in §5.1. Pass your own if you know what yours will look like.
- `fft_planning` and `fft_threads` change which grid is fastest on the processor, so pass
  what you will run with. The GPU ignores them.
- `gpu_acceleration = true` calibrates for the GPU, after `using CUDA`. Every trial and
  every timing then runs there, in your precision, since the fastest grid on a GPU is not
  the fastest on a processor.

## Reading the report

`calibrate` logs a report with `@info`, as in the output above, which Julia's standard
logging tools can silence.

- **Settings** gives ``M_G``, ``\Sigma/\Delta x`` and ``\lambda_\varepsilon``, the numbers
  that set the accuracy.
- **Grid** gives the grid chosen, with ``\Sigma/\sigma`` and ``R_c/\sigma``. In a dense
  suspension it can say *standard FCM* instead, which is explained under
  [the fastest grid](@ref "4. The fastest grid").
- **Measured error** is the error of these settings against the very accurate velocities.
  After it comes the time of one call, and the same speed in particle timesteps per second
  (the number of particles divided by the time of a call, §5.2).
- **Grids tried** lists every grid measured, with its ``\Sigma/\sigma``, its error and its
  fastest time. A grid marked *missed the tolerance* was not timed.

## What it does

`calibrate` repeats the procedure of §5 in four stages. Each stage only ever builds an
[`FFCMConfig`](@ref) and calls [`mobility!`](@ref), so what it measures is what you will
run.

### 1. A very accurate answer

An error has to be measured against something. `calibrate` computes fast FCM once, in double
precision whatever you are calibrating, with
[generous settings](@ref "The generous settings") aimed about a hundred times tighter than
``\varepsilon``. It does this at two Gaussian widths, the second on a grid 25 % finer,
because the exact answer does not depend on ``\Sigma/\sigma`` (§3). If the two differ by
more than ``\varepsilon/20``, `calibrate` stops and explains, as it does when two particles
share a position. In a box too small to allow anything wider than ``\sigma``, this
calculation simply is standard FCM.

When calibrating for the GPU, this calculation runs there too, still in double precision,
because even on consumer cards, where double precision is slow, that is far quicker than on
the processor. [`FFCMConfig`](@ref) then warns once that `Float64` is slow on such cards.
When you calibrate in `Float32`, that warning is about this calculation and not about your
runs. If the GPU is short of memory, both calculations move to a wider Gaussian on a coarser
grid, which takes more pair work but gives the same accuracy, and if even that does not fit
they run on the processor.

Standard FCM would be the simplest thing to compare with everywhere, since it needs no pair
correction. But its grid depends on the box and not on the number of particles: at
``\varepsilon = 10^{-4}`` a box of side 400 takes a grid of about 800³, which is 37 GB in
double precision, however few particles it holds. With 7,639 particles in that box, one call
took us 79 s on one thread, against about a second for both fast FCM calculations together,
on grids of 105³ and 135³.

### 2. The grid's accuracy

With the pair cutoff generous, so that the correction adds no error, `calibrate` finds the
smallest ``\Sigma/\Delta x`` whose error stays within ``\varepsilon/3``, as in §5.1, Fig.
1(a). During this search the stencil covers a generous number of Gaussian widths,
``M_G/(\Sigma/\Delta x)``. That is what cuts off the Gaussian, not ``M_G`` alone: at a fixed
``M_G``, the error turns back up as ``\Sigma/\Delta x`` grows. It then finds the smallest
``M_G`` at that ``\Sigma/\Delta x`` which adds at most another ``\varepsilon/3``. All the
trials use one Gaussian width, where [the starting grid](@ref "The starting grid") balances
grid and pair work for the generous settings. It is at least twice as wide as the
particle's, so that the correction is really at work.

### 3. The pair correction's accuracy

With the grid part generous, `calibrate` finds the smallest ``\lambda_\varepsilon`` whose
error stays within ``\varepsilon/3``, as in §5.2, Fig. 1(b).

These are three separate sources of error, so three thirds keep the total within
``\varepsilon`` even in the worst case. The measured total is usually about half of
``\varepsilon``. Cutting closer would save little, by our estimate about 10 % of the time of
a call, which isn't worth the risk of missing. Each search halves the remaining range at
every try, because the error only falls as a setting grows. Its steps, 0.02 in
``\Sigma/\Delta x`` and 0.1 in ``\lambda_\varepsilon``, each change the work by about 6 %.

### 4. The fastest grid

With those three numbers known, the grid fixes everything else:
``\Sigma = (\Sigma/\Delta x)\Delta x`` and ``R_c = \lambda_\varepsilon \Sigma``. A coarse
grid means a wide Gaussian and many pairs to correct, and a fine grid means the reverse, so
`calibrate` finds the balance by timing real calls (§5.2, Fig. 2):

- **Which grids.** Their sizes are made only of the factors 2, 3, 5 and 7, where FFTs are
  fastest. They have equal spacing along all axes (§3), buffers that fit in memory,
  ``R_c \leq \min(L)/2`` and ``M_G \leq \min(M)``. On a GPU, the grid's buffers may take
  60 % of its free memory after the particles' buffers, which leaves room for the FFT plans:
  on a card of 11 GB that allows a grid of 500³ in `Float32` and of 400³ in `Float64`, as we
  found fits there.
- **Where to look.** The walk begins at [the starting grid](@ref "The starting grid") and
  goes in both directions, in steps of at least 10 % of the grid size. Time against grid is
  one broad hump with a jagged top, so one slowdown does not mean the best has passed: in
  each direction the walk stops only after two grids in a row at least 30 % slower than the
  best so far.
- **What counts.** The error shifts with ``\Sigma/\sigma``, so every grid has its error
  measured first. A grid above ``0.95\varepsilon`` counts as infinitely slow, because the
  very accurate answer is only trusted to within ``\varepsilon/20``. So settings that miss
  the tolerance can't be returned.
- **Timing.** A grid's time is the median of at least three calls, after a short lead-in of
  untimed calls: a tenth of a second on a processor and a second on a GPU, which on small
  problems we found needs that long after a new grid is built to run at full speed again.
  Before the first grid, calls run untimed for a second on a processor and ten seconds on a
  GPU, so that the machine is warm.
- **The finish.** Every size within 10 % of the best is then compared, including sizes the
  walk skipped, because FFT speed varies between sizes. The grids take turns, because
  machines speed up and slow down, and taking turns gives each grid the same conditions.

Where this would widen the Gaussian by under 25 %, `calibrate` uses **standard FCM** instead
(``\Sigma = \sigma``). So little widening saves little grid work, yet it needs pair work
that standard FCM avoids, which is the point §5.2 makes for dense suspensions.

If no grid meets the tolerance, the targets of stages 2 and 3 are divided by three and the
stages repeat, at most three times. If that fails too, `calibrate` warns and returns the
very accurate settings, which are correct but slow. It never returns settings that failed.

### The generous settings

Each search needs one source of error to be negligible while it studies another, and the
very accurate answer needs all of them negligible. The paper does this with fixed, very
accurate runs (``R_c = L`` in §5.1 and ``\Sigma/\Delta x = 2`` in §5.2). Those would also
work here, but they cost up to ten times as much per trial, while the values below scale
with the accuracy asked for. **These formulas are ours and not the paper's.** Each estimates
how quickly one source of error decays at a target error ``e``, a hundredth of
``\varepsilon``, with a constant fitted to our measurements on suspensions of 1,000 to
64,000 particles:

| Setting | Generous value | Where its shape comes from |
|---|---|---|
| ``\Sigma/\Delta x`` | ``\sqrt{\ln(2.5/e)}/\pi`` | Spreading and interpolation together damp a Fourier mode by ``e^{-k^2\Sigma^2}``, which at the grid's highest mode ``k = \pi/\Delta x`` is ``e^{-\pi^2(\Sigma/\Delta x)^2}``. The measured error stays below about 2.5 times that once ``\Sigma/\sigma \geq 1.5``. |
| ``M_G/(\Sigma/\Delta x)`` | ``2\sqrt{2\ln(25/e)}``, with ``M_G`` rounded up to odd | With ``w`` widths on each side, the measured error is about ``25 e^{-w^2/2}``. |
| ``\lambda_\varepsilon`` | ``2\sqrt{\ln(1/e)}`` | The correction of equation (31) decays as ``e^{-r^2/4\Sigma^2}``, which at ``r = \lambda\Sigma`` is ``e^{-\lambda^2/4}``. This is close to measured values at tight tolerances and deliberately larger than them at loose ones. |

An estimate that proved too optimistic can't silently spoil a calibration: the very accurate
answer is checked at two widths, and every returned setting has its error measured against
it.

### The starting grid

This is **our own ansatz, not the paper's.** §5.2 says in words that the grid should balance
grid and pair costs, and that the best spacing shrinks as the particles get closer together.
The ansatz puts numbers to that, and its only job is to choose where the walk begins: the
measured times decide the answer.

1. Grid work per call grows with the number of grid points, ``a V/\Delta x^3``, ignoring the
   slow logarithm in the FFT, where ``V`` is the volume of the box and ``a`` is the cost of
   a grid point.
2. Each particle has ``(N/V)\tfrac{4}{3}\pi R_c^3`` neighbours, so there are
   ``\tfrac{2\pi}{3}(N^2/V) R_c^3`` pairs. With
   ``R_c = \lambda_\varepsilon (\Sigma/\Delta x) \Delta x``, pair work is
   ``b \tfrac{2\pi}{3}(N^2/V)(\lambda_\varepsilon \Sigma/\Delta x)^3 \Delta x^3``, where
   ``b`` is the cost of a pair.
3. A total of the form ``A/x + Bx``, here with ``x = \Delta x^3``, is smallest where its two
   parts are equal. That gives

   ```math
   \Delta x = \frac{(V/N)^{1/3}}{c^{1/6} \sqrt{\lambda_\varepsilon \Sigma/\Delta x}},
   \qquad c = \frac{2\pi}{3} \frac{b}{a}.
   ```

So the best spacing follows the distance between particles, ``(V/N)^{1/3}``. We use
``c = 30``, a rounded choice and not a measurement: on the processor we measured, the value
that fits lies between about 3 and 30. Because of the sixth root, being wrong by a factor of
ten moves the start by less than a factor of 1.5.

This beats starting from the coarsest grid, and the test suite holds it to that. We timed
each grid once and gave a walk from each start the same times, on one thread at
``\varepsilon = 10^{-4}``. The time includes building each grid, measuring its error and
timing it. Both walks found the same grid every time:

| Suspension | From the ansatz | From the coarsest grid | Saving |
|---|--:|--:|--:|
| ``\phi = 8\,\%``, 1,222 particles | 10 grids, 3.4 s | 14 grids, 3.7 s | 1.1× |
| ``\phi = 0.5\,\%``, 1,193 particles | 10 grids, 4.7 s | 14 grids, 6.4 s | 1.3× |
| ``\phi = 30\,\%``, 1,933 particles | 6 grids, 2.3 s | 12 grids, 4.6 s | 2.0× |
| ``\phi = 0.05\,\%``, 7,639 particles | 12 grids, 10 s | 21 grids, 46 s | 4.6× |

The saving grows with the number of particles, because the coarsest grids are the ones where
nearly every pair of particles has to be corrected.

### Odd stencil widths

The searches often land on an odd ``M_G``, and that is the efficient choice. The stencil is
[anchored at the nearest grid point](@ref "Stencil anchoring"), so an odd stencil is
symmetric about it, while an even one has one more point below than above and adds little
over the odd value below it. We measured this on a random suspension of 64,457 spheres
(``\phi = 8\,\%``, ``L = 150``), at ``\Sigma/\Delta x = 1.2`` and
``\Sigma/\sigma \approx 2``:

| ``M_G`` | 9 | 10 | 11 | 12 | 13 | 14 |
|---|--:|--:|--:|--:|--:|--:|
| Error | ``4.6 \times 10^{-3}`` | ``2.4 \times 10^{-3}`` | ``1.9 \times 10^{-4}`` | ``9.8 \times 10^{-5}`` | ``3.8 \times 10^{-6}`` | ``2.1 \times 10^{-6}`` |

Going from an odd width to the next even one about halves the error, while going from an
even width to the next odd one divides it by 13 to 26. The error still only falls as ``M_G``
grows, so the search needs no special case. The generous stencil is always odd for this
reason.

## What it costs

A calibration makes about twenty accuracy trials of one call each, then measures and times
about fifteen grids with several calls each. On the suspensions we tried, that came to about
the time of 200 to 300 calls at the settings it found. These are random suspensions in a box
of side 400, at ``\varepsilon = 10^{-4}`` in `Float32` on one thread, measured as described
under [Benchmarks](@ref "Calibration"):

| ``\phi`` | Particles | Calibration | Grid chosen | ``\Sigma/\sigma`` | One call | Measured error |
|---|--:|--:|--:|--:|--:|--:|
| 0.05 % | 7,639 | 23 s | 80³ | 9.0 | 84 ms | ``4.6 \times 10^{-5}`` |
| 0.2 % | 30,557 | 68 s | 120³ | 6.1 | 387 ms | ``4.7 \times 10^{-5}`` |
| 0.8 % | 122,230 | 6 min | 168³ | 4.5 | 1.6 s | ``4.7 \times 10^{-5}`` |
| 3.2 % | 488,923 | 20 min | 256³ | 2.9 | 5.9 s | ``6.8 \times 10^{-5}`` |

To check the measured errors, we also computed standard FCM on these suspensions without any
of `calibrate`'s code, on 800³ and 900³ grids that agreed to better than
``5 \times 10^{-7}``. Against that, every error above came out the same to the two figures
shown.

## Limits

- **The settings suit the positions given.** Calibrate again when the density, the
  clustering or ``N`` change a lot. Without `F`, the error refers to random forces.
- **Some requests are refused at once**, with a message saying why. The tolerance must lie
  between ``10^{-2}``, the loosest the paper covers (§5.1), and the point where rounding
  hides the error: ``5 \times 10^{-5}`` in `Float32`, ``10^{-10}`` in `Float64`. The box
  must hold the Gaussian, which takes a smallest side of about 5.6 sphere radii at
  ``\varepsilon = 10^{-2}``, 6.6 at ``10^{-4}`` and 8.9 at ``10^{-10}``. The sides of the
  box must allow an equal grid spacing, so they need a simple ratio.
- **Close contenders are close.** Near the best grid the times nearly tie, so two
  calibrations can choose different grids of about the same speed.
- **Settings found for one device suit that device.** ``M_G``, ``\Sigma/\Delta x`` and
  ``\lambda_\varepsilon`` only concern accuracy, but the fastest grid need not carry over
  from a processor to a GPU, or from one GPU to another, so calibrate on the device you will
  run.
- **On a GPU, timing has a fixed cost.** The untimed lead-ins add about ten seconds, plus a
  second for each grid timed, however quick the calls are, so even a small suspension takes
  about half a minute to calibrate there.

## Setting the numbers by hand

Given ``\Sigma/\Delta x``, ``M_G`` and ``\lambda_\varepsilon``, from a report or the table
below, each grid gives one set of settings. [`FFCMConfig`](@ref) checks what they must
satisfy: ``\Sigma \geq \sigma``, ``R_c \leq \min(L)/2``, ``M_G \leq \min(M)`` and equal
spacing.

```julia
σ = 1 / sqrt(π)              # the Gaussian width of a unit sphere
M = 96                       # the grid points per axis, which you choose
Δx = L_x / M
Σ = kernel_resolution * Δx   # kernel_resolution is Σ/Δx
settings = (;
    kernel_widths_ratio = Σ / σ,
    num_grid_points = Int32.((M, M, M)),
    M_G,
    R_c = cutoff_ratio * Σ,  # cutoff_ratio is λ_ε
)
```

These are the values `calibrate` found on the random suspension of 64,457 spheres above, as
a place to start, with their measured error at ``\Sigma/\sigma \approx 2``:

| ``\varepsilon`` | ``\Sigma/\Delta x`` | ``M_G`` | ``\lambda_\varepsilon`` | Error |
|---|--:|--:|--:|--:|
| ``10^{-2}`` | 0.78 | 7 | 3.2 | ``4.0 \times 10^{-3}`` |
| ``10^{-3}`` | 0.92 | 9 | 4.1 | ``3.9 \times 10^{-4}`` |
| ``10^{-4}`` | 1.04 | 11 | 5.5 | ``4.2 \times 10^{-5}`` |
| ``10^{-6}`` | 1.26 | 15 | 7.4 | ``4.1 \times 10^{-7}`` |
| ``10^{-8}`` | 1.44 | 19 | 8.7 | ``4.9 \times 10^{-9}`` |

Measure the error yourself before relying on these: it shifts with ``\Sigma/\sigma``, and at
``\varepsilon = 10^{-4}`` on this suspension the smallest ``\lambda_\varepsilon`` that
reaches ``\varepsilon`` itself grew from 4.4 at ``\Sigma/\sigma \approx 2`` to 5.8 at
``\Sigma/\sigma \approx 8``.
