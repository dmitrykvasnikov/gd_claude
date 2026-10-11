-- | Forward-mode automatic differentiation with dual numbers.
--
-- Gradient descent needs ∇L(θ), the vector of partial derivatives of the loss.
-- There are three ways to get it:
--
--   1. by hand: derive the formula on paper and code it (error prone, and must
--      be redone for every model and every loss);
--   2. numerically: (L(θ + h) - L(θ - h)) / 2h (approximate, and needs two
--      loss evaluations per parameter);
--   3. /automatically/: what this module does. It is exact and needs no
--      formulas.
--
-- The trick: instead of computing with a plain number x, compute with a pair
-- (x, x') holding the value /and/ its derivative. Every operation knows how
-- to update both parts using the rules from calculus:
--
-- >  (x, x') + (y, y') = (x + y,  x' + y')                sum rule
-- >  (x, x') · (y, y') = (x · y,  x'·y + x·y')            product rule
-- >  sin (x, x')       = (sin x,  cos x · x')             chain rule
--
-- Example: d/da of a·a + 3 at a = 2. Seed a as (2, 1), since da/da = 1:
--
-- >  (2,1)·(2,1) = (4, 1·2 + 2·1) = (4, 4)
-- >  (4,4) + (3,0) = (7, 4)                   value 7, derivative 4 = 2a ✓
--
-- With several parameters the derivative part becomes a vector, one slot per
-- parameter: θ₀ is seeded with (1,0,0,…), θ₁ with (0,1,0,…), and so on
-- ('variables'). After evaluating the loss once, the derivative part of the
-- result /is/ the gradient ∇L(θ). Constants such as the data (xᵢ, yᵢ) carry
-- an all-zero derivative, stored as an empty list.
module Math.AutoDiff
  ( Scalar (..)
  , Dual
  , primal
  , tangent
  , constant
  , variables
  , gradient
  ) where

import Math.Vector

-- | Numbers a model or loss can be computed with: plain Doubles or dual numbers.
class (Floating a, Ord a) => Scalar a where
  fromDouble :: Double -> a

-- | Plain doubles: the constant embedding is the identity.
instance Scalar Double where
  fromDouble = id

-- | A value together with its tangent (partial derivatives); a missing tail means zeros.
data Dual = Dual !Double [Double]
  deriving (Show)

-- | Dual numbers: constants have zero derivative.
instance Scalar Dual where
  fromDouble = constant

-- | Strict constructor that fully evaluates the tangent.
dual :: Double -> [Double] -> Dual
dual x dx = foldr seq () dx `seq` Dual x dx

-- | The value part of a dual number.
primal :: Dual -> Double
primal (Dual x _) = x

-- | The derivative part, padded with zeros to the requested length.
tangent :: Int -> Dual -> Vector Double
tangent n (Dual _ dx) = fromList (take n (dx ++ repeat 0))

-- | A dual number with zero derivative.
constant :: Double -> Dual
constant x = Dual x []

-- | Seed each component as an independent variable (derivative 1 w.r.t. itself).
variables :: Vector Double -> Vector Dual
variables v = fromList [Dual x (replicate i 0 ++ [1]) | (i, x) <- zip [0 ..] (toList v)]

-- | Value and gradient of a scalar function at a point.
gradient :: (Vector Dual -> Dual) -> Vector Double -> (Double, Vector Double)
gradient f v = (primal r, tangent (dim v) r)
  where
    r = f (variables v)

-- | Add two tangents, treating missing entries as zeros.
addT :: [Double] -> [Double] -> [Double]
addT [] ys = ys
addT xs [] = xs
addT (x : xs) (y : ys) = x + y : addT xs ys

-- | Scale a tangent by a constant.
scaleT :: Double -> [Double] -> [Double]
scaleT k = map (k *)

-- | Lift a unary function g and its derivative g' to dual numbers using the
-- chain rule: g(x, x') = (g x, g'(x) · x').
lift1 :: (Double -> Double) -> (Double -> Double) -> Dual -> Dual
lift1 f f' (Dual x dx) = dual (f x) (scaleT (f' x) dx)

-- | Dual numbers compare by value only.
instance Eq Dual where
  a == b = primal a == primal b

-- | Dual numbers are ordered by value only, so models may branch on comparisons.
instance Ord Dual where
  compare a b = compare (primal a) (primal b)

-- | Sum, difference and product rules.
instance Num Dual where
  Dual x dx + Dual y dy = dual (x + y) (addT dx dy)
  Dual x dx - Dual y dy = dual (x - y) (addT dx (scaleT (-1) dy))
  Dual x dx * Dual y dy = dual (x * y) (addT (scaleT y dx) (scaleT x dy))
  negate = lift1 negate (const (-1))
  abs = lift1 abs signum
  signum (Dual x _) = constant (signum x)
  fromInteger = constant . fromInteger

-- | Quotient rule.
instance Fractional Dual where
  Dual x dx / Dual y dy = dual (x / y) (addT (scaleT (1 / y) dx) (scaleT (-x / (y * y)) dy))
  recip = lift1 recip (\x -> -1 / (x * x))
  fromRational = constant . fromRational

-- | Derivatives of the elementary functions.
instance Floating Dual where
  pi = constant pi
  exp = lift1 exp exp
  log = lift1 log recip
  sqrt = lift1 sqrt (\x -> 1 / (2 * sqrt x))
  sin = lift1 sin cos
  cos = lift1 cos (negate . sin)
  tan = lift1 tan (\x -> 1 / (cos x * cos x))
  asin = lift1 asin (\x -> 1 / sqrt (1 - x * x))
  acos = lift1 acos (\x -> -1 / sqrt (1 - x * x))
  atan = lift1 atan (\x -> 1 / (1 + x * x))
  sinh = lift1 sinh cosh
  cosh = lift1 cosh sinh
  tanh = lift1 tanh (\x -> 1 - tanh x * tanh x)
  asinh = lift1 asinh (\x -> 1 / sqrt (x * x + 1))
  acosh = lift1 acosh (\x -> 1 / sqrt (x * x - 1))
  atanh = lift1 atanh (\x -> 1 / (1 - x * x))
  logBase b x = log x / log b
  Dual x dx ** Dual y dy = dual z (addT (scaleT (y * x ** (y - 1)) dx) exponentPart)
    where
      z = x ** y
      exponentPart
        | all (== 0) dy = []
        | otherwise = scaleT (z * log x) dy
