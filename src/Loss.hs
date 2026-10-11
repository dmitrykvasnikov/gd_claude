-- | Loss functions and the empirical risk that gradient descent minimises.
--
-- A /pointwise loss/ ℓ(p, y) says how bad a single prediction p is when the
-- true value is y. The /empirical risk/ averages it over the dataset:
--
-- >  L(θ) = (1/N) · Σᵢ ℓ(f(θ, xᵢ), yᵢ)
--
-- This one number is what GD and SGD push down. Its gradient follows from the
-- chain rule. For MSE, ℓ(p, y) = (p - y)², so
--
-- >  ∇L(θ) = (1/N) · Σᵢ 2·(f(θ, xᵢ) - yᵢ) · ∇_θ f(θ, xᵢ)
-- >                    └─── residual ──┘   └ how f reacts to θ ┘
--
-- Each sample pulls θ in the direction that reduces its own error, weighted by
-- how wrong it currently is. For @f(x) = a·x + b@ this gives
--
-- >  ∂L/∂a = (2/N) Σ (a·xᵢ + b - yᵢ)·xᵢ        ∂L/∂b = (2/N) Σ (a·xᵢ + b - yᵢ)
--
-- You never have to write these formulas yourself: 'lossGradient' evaluates
-- L on dual numbers (see "Math.AutoDiff") and gets the exact gradient out.
-- For SGD the same functions are simply called with a dataset of one sample
-- (or of one mini-batch) instead of the full data.
module Loss
  ( Loss (..)
  , mse
  , mae
  , huber
  , lossByName
  , isSquaredError
  , risk
  , lossValue
  , lossGradient
  ) where

import Data.Char (toLower)
import Dataset
import Math
import Model
import Text.Read (readMaybe)

-- | A pointwise loss ℓ(prediction, target). It is polymorphic in the number
-- type ('Scalar'), so the same code runs on Doubles (to get a value) and on
-- dual numbers (to get derivatives).
data Loss = Loss
  { lossName :: String
  , pointLoss :: forall a. Scalar a => a -> a -> a
  }

-- | Squared error (p - y)²; averaged it is the mean squared error. Large errors
-- are punished quadratically, so outliers have a strong pull. Its derivative
-- 2·(p - y) is proportional to the error, so steps get smaller near the optimum.
mse :: Loss
mse = Loss "MSE" (\p y -> (p - y) * (p - y))

-- | Absolute error |p - y|. Robust to outliers, but its derivative is always ±1,
-- so with a constant learning rate the steps do not shrink near the optimum
-- and the parameters keep bouncing around it (use --decay).
mae :: Loss
mae = Loss "MAE" (\p y -> abs (p - y))

-- | Huber loss: quadratic like MSE for errors below δ, linear like MAE beyond.
-- Smooth near the optimum, yet outliers cannot dominate.
huber :: Double -> Loss
huber delta = Loss ("Huber(δ=" ++ show delta ++ ")") f
  where
    f p y =
      let r = abs (p - y)
          d = fromDouble delta
       in if r <= d then r * r / 2 else d * (r - d / 2)

-- | Parse a loss from its command line name.
lossByName :: String -> Either String Loss
lossByName raw = case map toLower raw of
  "mse" -> Right mse
  "mae" -> Right mae
  "huber" -> Right (huber 1)
  'h' : 'u' : 'b' : 'e' : 'r' : ':' : d
    | Just delta <- readMaybe d, delta > 0 -> Right (huber delta)
  _ -> Left ("unknown loss '" ++ raw ++ "' (expected mse, mae, huber or huber:δ)")

-- | True for MSE, the loss the closed-form least squares solution minimises.
isSquaredError :: Loss -> Bool
isSquaredError l = lossName l == lossName mse

-- | Empirical risk: the average loss of the model over a dataset,
-- L(θ) = (1/N) Σᵢ ℓ(f(θ, xᵢ), yᵢ). Averaging (instead of summing) keeps the
-- scale of the gradient independent of N, so the same η works for a batch of
-- 1 sample, of 8 samples or of the full data.
risk :: Scalar a => Loss -> Model -> Dataset -> Vector a -> a
risk l m ds theta =
  average [pointLoss l (function m theta (fmap fromDouble (input s))) (fromDouble (target s)) | s <- ds]

-- | Empirical risk evaluated on plain doubles.
lossValue :: Loss -> Model -> Dataset -> Vector Double -> Double
lossValue = risk

-- | Empirical risk and its gradient ∇L(θ) with respect to the parameters,
-- computed in a single pass by automatic differentiation.
lossGradient :: Loss -> Model -> Dataset -> Vector Double -> (Double, Vector Double)
lossGradient l m ds = gradient (risk l m ds)
