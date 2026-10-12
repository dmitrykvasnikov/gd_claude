# descent: learn gradient descent by watching it

A small Haskell project that implements **gradient descent (GD)**, **stochastic
gradient descent (SGD)** and **mini-batch SGD** from scratch. You can watch them
work epoch by epoch in the terminal, pause between epochs, and compare them.

Everything is hand-written. The only dependency is `base`. That includes the
vectors and matrices, the linear solver, automatic differentiation, the random
number generator, and the tables and plots. The source is commented as a
tutorial. This README is the user manual plus a guided tour.

---

## Contents

1. [Build and run](#1-build-and-run)
2. [Gradient descent in five minutes](#2-gradient-descent-in-five-minutes)
3. [Reading the output](#3-reading-the-output)
4. [Guided experiments](#4-guided-experiments)
5. [Command line reference](#5-command-line-reference)
6. [Built-in models](#6-built-in-models)
7. [Using your own data (CSV)](#7-using-your-own-data-csv)
8. [Project structure and where to read the code](#8-project-structure-and-where-to-read-the-code)
9. [Adding your own model](#9-adding-your-own-model)
10. [Tests](#10-tests)

---

## 1. Build and run

You need GHC ≥ 9.10 and cabal. The project was built with GHC 9.10.3 and
cabal 3.16 from ghcup.

```sh
cabal build              # build library + executable
cabal run gd             # run with defaults: quadratic model, plain gradient descent
cabal run gd -- --help   # all options (everything after -- goes to the program)
cabal run gd -- --list   # the built-in models
cabal test               # run the test suite
```

To install the `gd` binary into `~/.local/bin`, run
`cabal install --installdir=$HOME/.local/bin`. The examples below write just
`gd …`, which means the same as `cabal run gd -- …`.

The most common invocations:

```sh
gd                                  # quadratic, gradient descent, table + plot
gd -M linear -m gd,sgd              # f(x) = a·x + b, GD vs SGD with a comparison table
gd -M linear -m sgd -w 5            # SGD, wait for a key press every 5 epochs
gd -M exp -m all -q                 # GD, SGD and mini-batch, only the final results
gd -M quadratic -x                  # also show the dataset and the matrix view
```

---

## 2. Gradient descent in five minutes

**The goal.** We have data points `(xᵢ, yᵢ)` and a model with unknown
parameters, e.g. `f(x) = a·x + b`. We want the parameters `θ = (a, b)` that
make the model fit the data. "Fit" is measured by the **loss**, here the mean
squared error:

```
L(θ) = (1/N) · Σᵢ (f(θ, xᵢ) − yᵢ)²
```

**The gradient.** `∇L(θ) = (∂L/∂a, ∂L/∂b)` points in the direction in which
the loss grows fastest. So we take a small step the *other* way:

```
θ ← θ − η · ∇L(θ)          η = learning rate (step size)
```

Repeat, and θ slides downhill into a minimum of the loss. For the line the
gradient is

```
∂L/∂a = (2/N) Σ (a·xᵢ + b − yᵢ) · xᵢ
∂L/∂b = (2/N) Σ (a·xᵢ + b − yᵢ)
```

Every point pulls the line towards itself, weighted by how wrong the
prediction currently is. As the fit improves the errors shrink, so the
gradient shrinks, so the steps shrink. Nobody has to tell the algorithm to
slow down.

**A tiny example by hand.** Fit `f(x) = a·x` to the single point `(1, 2)`,
starting at `a = 0` with `η = 0.1`. The loss is `(a − 2)²`, its derivative `2(a − 2)`:

| step | a | ∂L/∂a | new a = a − 0.1·∂L/∂a |
|---|---|---|---|
| 1 | 0 | −4 | 0.4 |
| 2 | 0.4 | −3.2 | 0.72 |
| 3 | 0.72 | −2.56 | 0.976 |
| … | → 2 | → 0 | |

**The learning rate matters.**

- Too small: progress is safe but slow.
- Just right: fast convergence.
- Too large: every step overshoots the bottom of the valley and lands higher
  on the other side. The parameters zig-zag, and the loss eventually explodes.

**GD vs SGD vs mini-batch.** They differ only in how much data is used per update:

| variant | gradient used for one update | updates per epoch | character |
|---|---|---|---|
| **GD** (full batch) | exact average over all N samples | 1 | smooth, predictable, expensive steps |
| **SGD** | one random sample | N | cheap, noisy steps, fast early progress, jitters near the end |
| **mini-batch** (B) | average of B random samples | N / B | the compromise used in practice |

An **epoch** is one pass over the whole dataset. SGD and mini-batch shuffle the
data at the start of every epoch.

**Why SGD works at all.** The gradient of one randomly chosen sample is
correct *on average*: its expected value equals the full gradient. Far from the
minimum all samples agree on the direction, so SGD races ahead with N updates
per epoch while GD does one. Near the minimum the samples disagree, and on noisy
data SGD keeps bouncing around the optimum. A decaying learning rate
(`--decay`) calms it down.

**Momentum** (`--momentum β`) keeps a running velocity `v ← β·v + ∇L`,
`θ ← θ − η·v`. Like a heavy ball rolling downhill, it speeds up along
consistent directions and damps zig-zagging.

The full explanation, with the stability rule for η and the role of the
condition number, is in the header comment of [`src/Descent.hs`](src/Descent.hs).

---

## 3. Reading the output

A run prints a **setup block**, an **epoch table** for every method, a
**summary**, and a **comparison** table when several methods ran.

```
 epoch │       loss   │     ‖∇L‖ │        η │             a             b
───────┼──────────────┼──────────┼──────────┼────────────────────────────
     0 │ 9.4046e+00   │ 7.05e+00 │ 1.00e-01 │      0.000000      0.000000
     5 │ 4.8293e-01 ↓ │ 1.54e+00 │ 1.00e-01 │      1.989509     -0.621868
    10 │ 3.2480e-02 ↓ │ 3.82e-01 │ 1.00e-01 │      2.392294     -0.865679
```

| column | meaning |
|---|---|
| `epoch` | passes over the data completed; epoch 0 is the starting point |
| `loss` | L(θ) on the **whole** dataset, also for SGD, so methods are comparable. `↓`/`↑` compares with the previous printed row |
| `‖∇L‖` | length of the full gradient. It goes to 0 at a minimum |
| `η` | learning rate used in that epoch (changes only with `--decay`) |
| `a`, `b`, … | current parameter values |

The **summary** after each run contains:

- `status`: one of
  - *converged*: loss ≤ `--tol`;
  - *finished*: the epoch limit was reached;
  - *diverged*: η is too large;
  - *stopped by user*: you pressed `q`.
- `epochs` and `parameter updates`: GD does 1 update per epoch, SGD does N.
- `final loss`, `fitted`: the learned formula with numbers filled in.
- `loss curve`: a sparkline of the loss on a log scale (each block is a
  factor, not a difference).
- A **parameter table**: initial, learned, the true values used to generate
  the data, and the exact **least squares** solution. The exact solution is
  available for models that are linear in their parameters, with MSE loss.
  `|Δ|` is how far GD/SGD ended from it (green < 1e-3 < yellow < 0.1 < red).
- An **ASCII plot** for one-input models: data `○`, fitted curve `•`, both in
  the same cell `◉`.

**Pausing.** With `-w N` the program stops after every N epochs:

| key | action |
|---|---|
| `enter` / `space` (any other key) | continue to the next pause |
| `c` | continue to the end without pausing |
| `q` | stop this run here (the summary is still printed; the next method still runs) |

Time spent waiting for a key is not counted in the reported run time.

---

## 4. Guided experiments

Each experiment below was run on this project. The numbers are what you
should see with the default seed.

**1. The learning rate.** Linear model, plain GD, different step sizes:

```sh
gd -M linear -q -r 0.01   # finished: 100 epochs not enough, still far away
gd -M linear -q -r 0.1    # converged in 64 epochs
gd -M linear -q -r 0.5    # converged in 17 epochs
gd -M linear -r 0.7 -p 1 -e 10 --no-plot   # watch `a` zig-zag: 0 → 4.79 → 0.34 → 4.56 …
gd -M linear -q -r 0.8    # diverged
```

At 0.7 the steps overshoot. `a` jumps from one side of the valley to the
other, and only slowly settles. A little more and the overshoot grows every
step.

**2. GD vs SGD on clean data.**

```sh
gd -M quadratic -m gd,sgd -q
```

GD needs ~200 epochs (200 updates). SGD needs ~27 epochs but makes 1080
updates. Count epochs (passes over the data) and SGD wins. Count updates and
GD wins. That is the trade-off.

**3. SGD on noisy data, and learning-rate decay.**

```sh
gd -M quadratic -m sgd --noise 0.3 -q               # ‖∇L‖ ≈ 0.24 at the end: still bouncing
gd -M quadratic -m sgd --noise 0.3 --decay 0.1 -q   # ‖∇L‖ ≈ 0.002: settles at the optimum
gd -M quadratic -m gd  --noise 0.3 -q               # GD reaches the least squares solution exactly
```

With noise no curve passes through all points. Each sample pulls in its own
direction, so constant-rate SGD never settles. Notice also that the learned
parameters match the **least squares** column, not the **truth** column. With
noisy data the best fit is not the generating curve.

**4. Measurement error: noise in y vs noise in x.**

Real measurements are imperfect in both coordinates. `--noise σ` blurs the
measured targets y, `--x-noise σ` blurs the measured inputs x. They look alike
in the plot but behave very differently:

```sh
gd -M linear -q --noise 0.5 -e 300                 # a = 2.521 (truth 2.5): scattered around the truth
gd -M linear -q --noise 0.5 -e 300 -n 2000         # a = 2.493: more data, closer to the truth
gd -M linear -q --x-noise 0.5 -e 300               # a = 2.062: the line is flatter
gd -M linear -q --x-noise 0.5 -e 300 -n 2000       # a = 2.074: more data does not help
gd -M linear -q --x-noise 0.5 -e 300 -n 2000 --range -6:6 -r 0.02   # a = 2.434: a wider x range does
```

Noise in y is random in both directions, so with more data it averages out
and the fit converges to the truth. Noise in x is not harmless in the same way.
The model is fed x̃ = x + error instead of the true x, and least squares
(which GD finds exactly, see the `|Δ|` column) then systematically
underestimates the slope by the factor

```
var(x) / (var(x) + σₓ²)        here (4/3) / (4/3 + 0.25) ≈ 0.84,  2.5 · 0.84 ≈ 2.1
```

This is called *regression dilution* or *attenuation bias*. It is not a
weakness of gradient descent. The loss itself prefers the wrong slope, so no
optimiser can recover the truth. You can only make the bias small by
measuring over a range much wider than the error (the last command:
var(x) = 12, factor ≈ 0.98). The wider range also makes the loss bowl steeper,
which is why it needs a smaller η.

**5. Ill-conditioning and momentum.**

```sh
gd -M cubic -q                              # 948 epochs
gd -M cubic -q --momentum 0.9 -r 0.01       # 263 epochs
gd -M cubic -x                              # look at XᵀX: entries range from ~2 to ~300
```

The features x³, x², x, 1 have very different scales, so the loss is a long
narrow valley. GD must use a small η (for the steep direction) and then crawls
along the flat one. Momentum builds up speed along the valley.

**6. Local minima (non-convex loss).**

```sh
gd -M sine -q                       # converges to a = 2, b = 1.5, c = 0.5
gd -M sine -q --init 1,3,0 -e 2000  # stuck: ‖∇L‖ ≈ 0 but loss ≈ 1.43
```

For `a·sin(b·x + c)` the loss has several valleys. A gradient of zero only
means "flat here", not "best possible". Where you start decides where you end.

**7. Loss functions.**

```sh
gd -M linear -m sgd -l mae -q               # MAE: derivative is always ±1, steps never shrink
gd -M linear -m sgd -l mae --decay 0.2 -q   # with decay it settles much closer
gd -M linear --noise 0.3 -l huber:0.5       # Huber: MSE near 0, MAE for large errors
```

**8. The linear algebra behind it.**

```sh
gd -M plane -x
```

This shows the design matrix X, the normal equations `XᵀX · θ = Xᵀy` and
their closed-form solution `θ* = (XᵀX)⁻¹Xᵀy`, the exact point gradient
descent is walking towards.

**9. Step by step.**

```sh
gd -M linear -m sgd -p 1 -w 1      # one epoch per key press
```

---

## 5. Command line reference

```
gd [OPTIONS]
```

### Model and data

| option | default | description |
|---|---|---|
| `-M`, `--model NAME` | `quadratic` | `linear`, `quadratic`, `cubic`, `poly:N` (1 ≤ N ≤ 30), `exp`, `sine`, `logistic`, `plane`. Aliases: `line`, `exponential`, `sin`, `sigmoid` |
| `-n`, `--samples N` | `40` | number of synthetic samples |
| `--noise σ` | `0` | measurement error in y: standard deviation of gaussian noise added to the targets |
| `--x-noise σ` | `0` | measurement error in x: standard deviation of gaussian noise added to every input |
| `--range LO:HI` | per model | input range of the synthetic data, e.g. `--range -3:3` |
| `--truth a,b,…` | per model | parameters used to generate the data |
| `--init a,b,…` | per model | starting parameters of descent |
| `-s`, `--seed N` | `42` | random seed (data generation and shuffling) |
| `-d`, `--data FILE` | | fit a CSV file instead of synthetic data (see [section 7](#7-using-your-own-data-csv)) |
| `-l`, `--loss NAME` | `mse` | `mse`, `mae`, `huber` (δ = 1) or `huber:δ` |

Numbers may be written with a leading dot (`.5`, `-.25`) everywhere, including
the lists of `--truth`, `--init` and the bounds of `--range`.

**How the synthetic data is made.** Each sample draws a true input x uniformly
from the range, computes the measured target `y = f(truth, x) + σ·ε` and
the measured input `x̃ = x + σₓ·δ` (ε, δ standard normal), and records
`(x̃, y)`. Both noise levels default to 0, so by default the data lies exactly
on the curve. For a fixed seed the true inputs and the target noise do not
depend on `--x-noise`, so runs with and without it are directly comparable.

### Algorithm

| option | default | description |
|---|---|---|
| `-m`, `--method LIST` | `gd` | comma separated list of `gd`, `sgd`, `mb:N` (mini-batch of N), or `all` (= `gd,sgd,mb:8`). Each method runs on the same data from the same start |
| `-r`, `--rate η` | per model and method | learning rate. Without it, GD and SGD get separately tuned defaults |
| `-e`, `--epochs N` | per model | maximum number of epochs |
| `--decay D` | `0` | learning-rate schedule `η / (1 + D·epoch)` |
| `--momentum β` | `0` | classical momentum, `0 ≤ β < 1` |
| `--tol ε` | `1e-12` | stop as soon as the loss is ≤ ε |

### Output and interaction

| option | default | description |
|---|---|---|
| `-p`, `--print-every N` | about 20 rows per run | print every N-th epoch; `0` = no epoch table. Epoch 0 and the last epoch are always printed |
| `-w`, `--pause-every N` | `0` (never) | wait for a key press every N epochs |
| `-q`, `--quiet` | | no epoch table and no plot, only the summaries |
| `-x`, `--matrices` | | show the dataset table, the design matrix X and the normal equations |
| `--no-plot` | | skip the ASCII plot |
| `--color` / `--no-color` | auto | colors are on when the output is a terminal |
| `--list` | | list the built-in models |
| `-h`, `--help` | | show help |

Options can be combined freely. For example, `-q -w 50` prints nothing but
still stops every 50 epochs, showing the row of that epoch.

---

## 6. Built-in models

| name | formula | what it teaches |
|---|---|---|
| `linear` | `f(x) = a·x + b` | the basic case: convex, fast |
| `quadratic` | `f(x) = a·x² + b·x + c` | nonlinear in x but linear in θ, still convex |
| `cubic` | `f(x) = a·x³ + b·x² + c·x + d` | ill-conditioning, momentum helps |
| `poly:N` | `a·xᴺ + … ` | conditioning gets worse quickly with the degree |
| `exp` | `f(x) = a·e^(b·x)` | nonlinear in a parameter, curvature changes along the way |
| `sine` | `f(x) = a·sin(b·x + c)` | non-convex: local minima |
| `logistic` | `f(x) = a / (1 + e^(−b·(x − c)))` | plateaus with tiny gradients |
| `plane` | `f(x, y) = a·x + b·y + c` | two inputs (no plot, but the matrix view works) |

Every model comes with true parameters for the synthetic data, a starting
point, an input range, an epoch limit and two default learning rates (GD and
SGD), all overridable from the command line.

---

## 7. Using your own data (CSV)

```sh
gd -M linear -d points.csv -m gd,sgd
```

- One sample per line: the inputs, then the target as the last column.
  `x,y` for one-input models, `x,y,z` for `plane`.
- Separators: comma, semicolon or tab.
- Blank lines and lines starting with `#` are ignored, and so is a non-numeric
  first line (a header).
- Numbers like `.5`, `-.5`, `+1`, `2.` and `1e-3` are accepted.

```csv
# measurements
x,y
0,1
1,3.1
2,4.9
3,7.2
```

With a CSV file there is no "truth" column. For linear-in-θ models with MSE
the least squares column is still shown. `--noise` and `--x-noise` only apply
to synthetic data and are ignored with `-d`.

---

## 8. Project structure and where to read the code

```
src/
  Math.hs              re-exports the three Math modules
  Math/Vector.hs       vectors: +, −, scaling, dot product, norm
  Math/Matrix.hs       matrices: products, transpose, Gaussian elimination
  Math/AutoDiff.hs     automatic differentiation with dual numbers
  Model.hs             the model abstraction f(θ, x)
  Model/Catalog.hs     built-in models and their example problems
  Loss.hs              MSE / MAE / Huber and the empirical risk L(θ)
  Dataset.hs           synthetic data with measurement noise, CSV parsing, batching
  LeastSquares.hs      normal equations: the exact answer for linear models
  Descent.hs           ★ GD, SGD, mini-batch: the algorithm itself
  Rng.hs               random numbers, gaussian noise, shuffling
  Pretty.hs            number formats, matrices, tables, sparkline, plot
  Report.hs            epoch rows, summaries, comparison tables
  Runner.hs            prints epochs and waits for key presses
app/
  Main.hs, Options.hs  command line program
test/
  Spec.hs              test suite
```

Suggested reading order for learning:

1. **`src/Descent.hs`**: the long header comment explains GD, SGD,
   mini-batches, learning rate, momentum and stopping. Then `step` (one
   update), `batchesFor` (what makes GD, SGD and mini-batch different) and
   `descend` (the loop).
2. **`src/Loss.hs`**: what is being minimised and what its gradient looks like.
3. **`src/Math/AutoDiff.hs`**: how the gradient is computed without writing
   any derivative by hand.
4. **`src/Model.hs`** and **`src/Model/Catalog.hs`**: how models are described.
5. **`src/LeastSquares.hs`**: the closed-form solution and why GD converges to it.

Design notes:

- **Training is a pure lazy stream.** `descend` returns `[Epoch]`. The
  algorithm knows nothing about printing or pausing. `Runner` consumes the
  list one epoch at a time, and since the list is lazy, the next epoch is only
  computed when it is requested.
- **One model definition, two uses.** A model is a polymorphic function. Run
  on `Double` it predicts. Run on dual numbers it yields the exact gradient.
- **Loss and gradient are always reported on the full dataset**, even for SGD,
  so the numbers of different methods can be compared directly.

---

## 9. Adding your own model

Write the formula once. The gradient comes for free. For example, a gaussian
bump `a·e^(−(x − b)² / c)`:

```haskell
-- in src/Model/Catalog.hs
gaussian :: Model
gaussian =
  mkModel
    "gaussian bump"                       -- name
    ["a", "b", "c"]                       -- parameter names
    ["x"]                                 -- input names
    (\ps -> concat [ps !! 0, "·e^(-(x - ", ps !! 1, ")² / ", ps !! 2, ")"])  -- pretty printer
    (\t x -> t ! 0 * exp (negate ((x ! 0 - t ! 1) ^ 2) / t ! 2))            -- f(θ, x)
```

Then add an entry to `examples` with the true parameters, start, input range,
GD rate, SGD rate, epoch limit and a description:

```haskell
  , Example "gauss" gaussian (fromList [2, 0.5, 1]) (fromList [1, 0, 2]) (-3, 3) 0.1 0.02 1000 "bell curve"
```

It is now available as `gd -M gauss`.

If the model is **linear in its parameters**, use `linearModel` and give the
feature vector φ(x) instead. You then also get the closed-form least squares
column and the `-x` matrix view:

```haskell
-- f(x) = a·sin x + b·cos x
waves :: Model
waves = linearModel "waves" ["a", "b"] ["x"]
  (\ps -> sumTerms [ps !! 0 ++ "·sin x", ps !! 1 ++ "·cos x"])
  (\x -> fromList [sin (x ! 0), cos (x ! 0)])
```

---

## 10. Tests

```sh
cabal test
```

51 checks cover:

- **linear algebra**: products, transpose, solving with pivoting, singular
  matrices;
- **automatic differentiation**, compared against finite differences;
- **the random generator and shuffling**;
- **measurement noise**: x-noise only moves the inputs, has the requested
  spread, and flattens the least squares line by the predicted factor;
- **CSV parsing and output formatting**;
- **the algorithms themselves**:
  - GD, SGD and mini-batch recover the true parameters on clean data;
  - on noisy data GD reaches the least squares solution;
  - GD's loss never increases with a small η;
  - a huge η is detected as divergence;
  - momentum converges.
