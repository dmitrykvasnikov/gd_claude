# AGENTS.md

Guidance for coding agents working on this repository.

## What this is

`descent` is a teaching project: gradient descent (GD), stochastic gradient
descent (SGD) and mini-batch SGD implemented from scratch in Haskell, with a
terminal UI that prints epochs, can pause between them, and compares methods.
The original brief is in `prompt.md`; the user manual and guided tour is
`README.md`.

## Hard constraints (from `prompt.md`)

- **No math / matrix libraries.** The only dependency is `base`. Vectors,
  matrices, the linear solver, autodiff, RNG, tables and plots are all
  hand-written. Do not add `vector`, `hmatrix`, `random`, `ad`, etc.
- All math lives under `src/Math/` (re-exported by `src/Math.hs`).
- The `Model` abstraction must stay general enough for arbitrary
  differentiable `f(θ, x)`, not just polynomials.
- Per-epoch printing and pausing are optional; running to the final result
  without interaction must always work (`-q`, `-p 0`, `-w 0`).
- Every top-level function has a short `-- |` Haddock comment. Module headers
  in `Descent`, `Loss`, `Model`, `Math.AutoDiff`, `LeastSquares` are written as
  tutorials; keep that tone when editing them.

## Toolchain and commands

GHC 9.10.3, cabal 3.16 (ghcup). There is **no** `stack.yaml`; use cabal.
`base >= 4.20` is required (Prelude exports `foldl'`, which the code relies on).

```sh
cabal build                    # library + `gd` executable (-Wall -O2)
cabal test                     # custom test runner, prints ✓/✗ per check
cabal run gd -- --help
cabal run gd -- -M linear -m all -q --noise 0.2
$(cabal list-bin gd) …         # call the built binary directly
```

The build must stay warning-free under `-Wall`. Default language is `GHC2021`
(gives `RankNTypes`, needed for the polymorphic `function`/`pointLoss` fields).

## Layout

```
src/Math/Vector.hs     list-backed Vector a, ^+^ ^-^ *^ dot norm force …
src/Math/Matrix.hs     list-of-rows Matrix a, |*| |*^ ^*|, gram, solve (Gaussian elim. w/ pivoting)
src/Math/AutoDiff.hs   forward-mode dual numbers; class Scalar (Double | Dual); gradient
src/Rng.hs             SplitMix64 Gen, uniform, uniformR, normal (Box–Muller), draws, shuffle
src/Model.hs           Model record, mkModel (general), linearModel (θ·φ(x)), predict, rendering helpers
src/Model/Catalog.hs   built-in models + Example records (truth, start, domain, GD/SGD rates, epochs)
src/Loss.hs            Loss (MSE/MAE/Huber), risk L(θ), lossGradient via autodiff
src/Dataset.hs         Sample, synthesize (synthetic data + noise), parseCsv, chunksOf
src/LeastSquares.hs    normal equations XᵀXθ = Xᵀy, closed-form reference for linear-in-θ + MSE
src/Descent.hs         Method, Config, Epoch, step, batchesFor, descend (lazy [Epoch]), outcome
src/Pretty.hs          generic ANSI/table/matrix/sparkline/plot rendering
src/Report.hs          domain rendering: epoch rows, run summary, comparison, matrix views
src/Runner.hs          IO driver: consumes [Epoch], prints, pauses (raw-mode key read)
app/Options.hs         GetOpt CLI parsing and validation
app/Main.hs            wiring: example lookup, data source, configs, output
test/Spec.hs           all tests (a flat list of `Test name Bool`)
```

Dependency direction: `Math` ← `Rng`, `Model` ← `Dataset` ← `Loss` ←
`Descent` ← `Model.Catalog`, `Report` ← `Runner` ← `app/`. `Pretty` depends only
on `Math`. Keep the algorithm modules (`Descent`, `Loss`, `Dataset`) pure; all
IO stays in `Runner` and `app/`.

## Key design invariants

- **Training is a pure lazy stream.** `descend :: Config -> Model -> Dataset ->
  Vector Double -> [Epoch]`; epoch 0 is the start point. `Runner` decides how
  much of it to consume; pausing works because the next epoch is not computed
  until requested. Don't make `descend` strict over the whole list or move IO
  into it.
- **One polymorphic model definition** (`forall a. Scalar a => Vector a ->
  Vector a -> a`) serves both evaluation (`Double`) and differentiation
  (`Dual`). Never hand-write gradients. Models must only use `Floating`/`Ord`
  operations and `fromDouble` for constants coming from data.
- **Loss and gradient in `Epoch` are always on the full dataset**, also for
  SGD/mini-batch, so methods are comparable. Batch gradients are internal.
- `risk` *averages* (not sums) the pointwise loss, so the same η works for any
  batch size.
- Strictness: `force` vectors in loops (see `step`), `Dual` tangents are forced
  by the smart constructor `dual`. Keep this when adding operations or you get
  thunk leaks on long runs.
- Least squares reference only exists when `features m` is `Just` and the loss
  is MSE (`isSquaredError` compares by name).
- Divergence = loss NaN/Inf or > 1e12·(1 + initial loss).

## Data generation and measurement error

`Dataset.synthesize m θ (lo, hi) n σ σₓ gen` produces `n` samples imitating
noisy measurements:

```
x  ~ Uniform[lo, hi)                 true input (per input dimension)
y  = f(θ, x) + σ·ε,   ε ~ N(0, 1)    measured target   (--noise σ)
x̃ = x + σₓ·δ,        δ ~ N(0, 1)    measured input    (--x-noise σ)
```

and records `(x̃, y)`, sorted by x̃. Both default to 0, so **default data is
noise-free**; pass the flags to imitate measurement error.

- Noise is additive, Gaussian and homoscedastic (same absolute σ everywhere).
  There is no relative (σ·|y|) noise and no outlier model.
- RNG stream layout matters, keep it when editing `synthesize`:
  - per sample: `inputDim` uniforms for x, then one normal for ε. The ε draw
    happens even when σ = 0, so x does not depend on σ;
  - the x-noise δ is drawn from the generator left *after* all samples, so x and
    ε do not depend on σₓ, and with σₓ = 0 the data is byte-identical to the
    pre-x-noise implementation (README numbers stay valid).
- Data uses `mkGen seed`; shuffling uses `seed + 1` (set in `app/Main.hs`).
- With y-noise, loss never reaches `--tol`, so runs end as "finished" at the
  epoch limit, and the learned θ matches least squares, not the truth.
- x-noise causes *attenuation bias*: the least-squares slope shrinks by
  var(x) / (var(x) + σₓ²) and does not go away with more samples. This is
  expected and tested (`regressionDilution` in `test/Spec.hs`), not a bug.
- Both noise flags are ignored when `-d FILE` is used.

## Common tasks

**Add a model:** define it in `src/Model/Catalog.hs` with `mkModel` (general)
or `linearModel` (gives least squares + `-x` view), add an `Example` to
`examples` (key, model, truth, start, domain, GD rate, SGD rate, epochs, blurb),
export it, update README sections 5–6, and add a recovery test to `test/Spec.hs`.
Default rates are tuned by experiment: check GD and SGD both converge on
clean data and don't diverge.

**Add a CLI option:** add a field + default in `app/Options.hs`, an `Option`
entry using the validators (`positive`, `nonNegative`, `fraction`, `natural`,
`natural0`, `integer`, `numbers`; all go through `readNumber`, which accepts
`.5`/`-.5`), wire it in `app/Main.hs`, document it in the
README command-line reference and the `usage` examples if notable.

**Add a test:** append `Test "name" bool` to `tests` in `test/Spec.hs`; use
`near`/`nearV`/`nearM`, `dataset ex σ` (fixed seed 7, 40 samples), `fit`,
`recovers`. Update the check count in README section 10 (currently 51).
`app/Options.hs` is not part of the library, so CLI parsing is verified by
running the binary, not by `Spec.hs`.

## Keep README in sync

README section 4 ("Guided experiments") quotes concrete numbers (epochs,
‖∇L‖, losses) produced with the default seed 42. If you change RNG, data
generation, default rates/epochs, or the update rule, re-run those commands
and update the numbers.

## Known rough edges

- `polyExample` blurb always says "(here N = 4)".
- `Vector` / `Matrix` are lists: `!` is O(n), fine for the small sizes here
  but don't use them for large P or N.
- Noise is homoscedastic only; no relative or outlier noise (see above).
