-- | Gradient descent, stochastic gradient descent and mini-batch SGD.
--
-- = The problem
--
-- We have a model @f(θ, x)@ with parameters @θ = (a, b, …)@ and a dataset of
-- N samples @(xᵢ, yᵢ)@. We want the θ for which the model fits the data best,
-- i.e. the θ that minimises the /loss/ (empirical risk, see "Loss"):
--
-- >  L(θ) = (1/N) · Σᵢ ℓ(f(θ, xᵢ), yᵢ)        e.g. ℓ(p, y) = (p - y)²  (MSE)
--
-- = The idea of gradient descent
--
-- The gradient @∇L(θ) = (∂L/∂a, ∂L/∂b, …)@ is a vector that points in the
-- direction in which L grows fastest. Walking a small step in the /opposite/
-- direction makes L smaller. Repeat that and you slide down into a minimum:
--
-- >  θ ← θ - η · ∇L(θ)
--
-- η (eta) is the /learning rate/, the step size:
--
--   * too small: every step is safe, but you need very many of them;
--   * too large: you jump over the valley, land higher on the other side, and
--     the loss grows every step: the run /diverges/ (try @gd -M linear -r 3@).
--
-- For MSE the loss is a quadratic bowl and the rule is precise: GD is stable
-- when η < 2 / λ_max, where λ_max is the largest curvature (eigenvalue of the
-- Hessian) of L. The ratio between the steepest and the flattest direction
-- (the /condition number/) decides how slow the flat directions converge. This
-- is why the cubic and high degree polynomials need so many epochs.
--
-- = A tiny worked example
--
-- Model @f(x) = a·x@, one sample @(x, y) = (1, 2)@, MSE, start @a = 0@, @η = 0.1@:
--
-- >  L(a)  = (a·1 - 2)²            ∂L/∂a = 2·(a - 2)·1
-- >  a = 0     → ∂L/∂a = -4    → a ← 0    - 0.1·(-4)   = 0.4
-- >  a = 0.4   → ∂L/∂a = -3.2  → a ← 0.4  - 0.1·(-3.2) = 0.72
-- >  a = 0.72  → ∂L/∂a = -2.56 → a ← 0.72 - 0.1·(-2.56) = 0.976   …  → 2
--
-- The gradient shrinks as we approach the minimum, so the steps shrink too.
-- Every step removes 20 % of the remaining distance to a = 2.
--
-- = GD, SGD and mini-batches
--
-- The three variants differ only in /how much data/ is used to estimate the
-- gradient for one update:
--
--   * __Gradient descent (full batch)__: the exact gradient over all N
--     samples. One update per pass over the data. Smooth and predictable, but
--     every single step costs N model evaluations.
--
--   * __Stochastic gradient descent__: the gradient of /one/ random sample,
--     @∇ℓ(f(θ, xᵢ), yᵢ)@. It is a noisy estimate, but an /unbiased/ one: if
--     the sample is picked uniformly at random, its expected value is exactly
--     ∇L(θ). You get N cheap updates per pass. Far from the minimum all
--     samples roughly agree on the direction, so SGD makes fast progress.
--     Near the minimum they disagree, so with noisy data the parameters keep
--     jittering around the optimum instead of settling. A decaying learning
--     rate (@--decay@) shrinks that jitter. A single sample can also be much
--     steeper than the average, which is why SGD needs a smaller η than GD.
--
--   * __Mini-batch SGD__: the average gradient of B random samples. It is the
--     compromise: less noise than SGD (variance shrinks like 1/B), more
--     updates per pass than GD (N/B of them).
--
-- One full pass over the data is an /epoch/. For the stochastic variants the
-- data is shuffled at the start of every epoch, so each sample is used exactly
-- once per epoch, but in a different random order every time.
--
-- = Momentum (optional)
--
-- With @--momentum β@ the update keeps a running "velocity":
--
-- >  v ← β·v + ∇L          θ ← θ - η·v
--
-- Like a heavy ball, it keeps rolling in directions where the gradients agree
-- and damps zig-zagging where they alternate. β = 0 is plain (S)GD.
--
-- = How this module is organised
--
-- 'descend' produces a /lazy list/ of 'Epoch's, epoch 0 being the starting
-- point. Nothing is computed until somebody looks at the list, so the caller
-- (see "Runner") decides how much to consume, what to print and when to
-- pause, while the algorithm itself stays pure and easy to test.
module Descent
  ( Method (..)
  , Config (..)
  , Epoch (..)
  , Outcome (..)
  , Run (..)
  , methodName
  , methodTag
  , rateAt
  , step
  , batchesFor
  , descend
  , outcome
  , finalEpoch
  ) where

import Data.Bifunctor (first)
import Dataset
import Loss
import Math
import Model
import Rng

-- | Which variant to run: full batch (GD), one sample per update (SGD), or
-- B samples per update (mini-batch SGD).
data Method
  = FullBatch
  | Stochastic
  | MiniBatch Int
  deriving (Eq, Show)

-- | Hyper-parameters of a training run: settings chosen by you, not learned
-- from the data.
data Config = Config
  { method :: Method
  -- ^ GD, SGD or mini-batch
  , learningRate :: Double
  -- ^ η, the step size at epoch 0
  , decay :: Double
  -- ^ d in the schedule η/(1 + d·epoch); 0 keeps η constant
  , momentum :: Double
  -- ^ β, how much of the previous velocity is kept; 0 disables momentum
  , maxEpochs :: Int
  -- ^ stop after this many passes over the data
  , tolerance :: Double
  -- ^ stop early as soon as the loss is at most this small
  , seed :: Int
  -- ^ seed for shuffling the data (SGD and mini-batch)
  , objective :: Loss
  -- ^ the pointwise loss ℓ that defines L(θ)
  }

-- | Snapshot taken at the end of an epoch (epoch 0 is the starting point).
-- The loss and gradient are always measured on the /whole/ dataset, also for
-- SGD, so that the numbers of different methods can be compared.
data Epoch = Epoch
  { epochNo :: Int
  -- ^ how many passes over the data have been completed
  , epochParams :: Vector Double
  -- ^ θ after this epoch
  , epochLoss :: Double
  -- ^ L(θ) on the full dataset
  , epochGrad :: Vector Double
  -- ^ ∇L(θ) on the full dataset; its length goes to 0 at a minimum
  , epochRate :: Double
  -- ^ learning rate used during this epoch
  , epochUpdates :: Int
  -- ^ total number of parameter updates so far (GD: 1 per epoch, SGD: N per epoch)
  }

-- | How a run ended.
data Outcome
  = Converged
  -- ^ the loss dropped to the tolerance
  | Exhausted
  -- ^ the epoch limit was reached first
  | Diverged
  -- ^ the loss blew up, the learning rate is too large
  | Interrupted
  -- ^ the user pressed q
  deriving (Eq, Show)

-- | A finished run: its configuration, every epoch, the outcome and the time
-- spent computing (time spent waiting for key presses is excluded).
data Run = Run
  { runConfig :: Config
  , runHistory :: [Epoch]
  , runOutcome :: Outcome
  , runSeconds :: Double
  }

-- | Human readable name of a method.
methodName :: Method -> String
methodName FullBatch = "Gradient descent (full batch)"
methodName Stochastic = "Stochastic gradient descent (batch 1)"
methodName (MiniBatch b) = "Mini-batch SGD (batch " ++ show b ++ ")"

-- | Short tag of a method for tables.
methodTag :: Method -> String
methodTag FullBatch = "GD"
methodTag Stochastic = "SGD"
methodTag (MiniBatch b) = "MB-" ++ show b

-- | Learning rate for epoch k with inverse-time decay: η_k = η / (1 + d·k).
-- Large steps early to make fast progress, smaller steps later to settle down,
-- which matters for SGD on noisy data. With d = 0 the rate is constant.
rateAt :: Config -> Int -> Double
rateAt cfg k = learningRate cfg / (1 + decay cfg * fromIntegral k)

-- | One parameter update, the heart of every variant.
--
-- >  v' = β·v + g            (with β = 0 this is just v' = g)
-- >  θ' = θ - η·v'
--
-- @g@ is a gradient estimate: the exact full-data gradient for GD, or the
-- gradient of a single sample / mini-batch for SGD.
step :: Double -> Double -> (Vector Double, Vector Double) -> Vector Double -> (Vector Double, Vector Double)
step beta eta (theta, velocity) g =
  let velocity' = force (beta *^ velocity ^+^ g)
      theta' = force (theta ^-^ eta *^ velocity')
   in (theta', velocity')

-- | Split the dataset into the batches used during one epoch.
--
--   * GD: a single batch holding every sample, so one update per epoch.
--   * SGD: shuffle, then N batches of one sample each, so N updates.
--   * Mini-batch: shuffle, then batches of B samples, so ⌈N/B⌉ updates.
--
-- Shuffling every epoch matters: going through the data in a fixed order
-- (here it is even sorted by x) would make consecutive updates pull in
-- correlated directions instead of behaving like independent random samples.
batchesFor :: Method -> Dataset -> Gen -> ([Dataset], Gen)
batchesFor FullBatch ds g = ([ds], g)
batchesFor Stochastic ds g = first (chunksOf 1) (shuffle ds g)
batchesFor (MiniBatch b) ds g = first (chunksOf b) (shuffle ds g)

-- | The training loop, as a lazy list of epochs starting with epoch 0 (the
-- initial parameters). The list ends at the epoch limit, once
-- loss ≤ tolerance, or when the loss explodes.
--
-- In pseudo code:
--
-- > θ ← θ₀ ; v ← 0
-- > repeat for every epoch k:
-- >     record (k, θ, L(θ), ∇L(θ))                 -- measured on all data
-- >     stop if k = maxEpochs, L ≤ tolerance or L exploded
-- >     for each batch B in batchesFor(method, data):
-- >         g ← ∇L_B(θ)                            -- gradient on this batch only
-- >         (θ, v) ← step(β, η_k, (θ, v), g)
descend :: Config -> Model -> Dataset -> Vector Double -> [Epoch]
descend cfg m ds theta0 = go 0 (force theta0) (zeros (dim theta0)) (mkGen (seed cfg)) 0
  where
    loss = objective cfg
    initialLoss = lossValue loss m ds theta0

    -- Record the state at the start of epoch k, then (unless we are done)
    -- run the epoch. The recursive call sits behind the ':' and is therefore
    -- only evaluated when the consumer asks for the next epoch.
    go k theta velocity g updates =
      let (l, fullGrad) = lossGradient loss m ds theta
          here = Epoch k theta l fullGrad (rateAt cfg (max 0 (k - 1))) updates
       in here : if finished k l then [] else runEpoch k theta fullGrad velocity g updates

    finished k l = k >= maxEpochs cfg || exploded initialLoss l || l <= tolerance cfg

    -- One epoch: walk through the batches, updating θ after each of them.
    runEpoch k theta fullGrad velocity g updates =
      let eta = rateAt cfg k
          (batches, g') = batchesFor (method cfg) ds g
          update (th, v) batch = step (momentum cfg) eta (th, v) (batchGradient batch th)
          (theta', velocity') = case method cfg of
            -- GD: the single batch is the whole dataset, whose gradient we
            -- have just computed for the epoch record, so reuse it.
            FullBatch -> step (momentum cfg) eta (theta, velocity) fullGrad
            -- SGD / mini-batch: each update uses the gradient of its own
            -- batch, evaluated at the parameters left by the previous update.
            _ -> foldl' update (theta, velocity) batches
       in go (k + 1) theta' velocity' g' (updates + length batches)

    -- Gradient of the loss restricted to one batch: (1/|B|) Σ_{i∈B} ∇ℓᵢ(θ).
    batchGradient batch th = snd (lossGradient loss m batch th)

-- | The loss is not a finite number anymore, or it has grown by a factor of
-- 10¹² compared to the start. Either way the steps overshoot more and more,
-- which only a smaller learning rate can fix.
exploded :: Double -> Double -> Bool
exploded initial l = isNaN l || isInfinite l || l > 1e12 * (1 + initial)

-- | Classify how a run ended, given the initial loss and the last epoch.
outcome :: Config -> Double -> Epoch -> Outcome
outcome cfg initial e
  | exploded initial l = Diverged
  | l <= tolerance cfg = Converged
  | epochNo e >= maxEpochs cfg = Exhausted
  | otherwise = Interrupted
  where
    l = epochLoss e

-- | Last epoch of a run, if any.
finalEpoch :: Run -> Maybe Epoch
finalEpoch r = case runHistory r of
  [] -> Nothing
  es -> Just (last es)
