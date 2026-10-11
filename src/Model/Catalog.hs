-- | Ready-made models and example problems (true parameters, starting point, defaults).
--
-- Each example shows a different aspect of gradient descent:
--
--   * linear, plane: convex and well conditioned, so GD converges in a few dozen epochs.
--   * quadratic, cubic, poly:N: still convex (linear in θ), but the features
--     x, x², x³ … have very different scales and are correlated, so the loss
--     bowl is a long narrow valley. GD zig-zags across it and crawls along it,
--     and higher degrees need many more epochs. Momentum helps here.
--   * exp: nonlinear in b. The curvature changes along the way, so a rate
--     that is fine at the start can oscillate near the end.
--   * sine: non-convex, with many valleys. Start with b far from the truth
--     (@--init 1,3,0@) and GD settles in a wrong local minimum.
--   * logistic: nonlinear in all parameters, with flat plateaus where the
--     gradient is tiny.
--
-- The default rates were found by experiment. GD tolerates larger steps than
-- SGD, because a single sample's gradient can be much steeper than the average.
module Model.Catalog
  ( Example (..)
  , examples
  , findExample
  , defaultRate
  , linear
  , quadratic
  , cubic
  , polynomial
  , exponential
  , sine
  , logistic
  , plane
  ) where

import Data.Char (isDigit, toLower)
import Descent (Method (..))
import Math
import Model

-- | A model together with a synthetic problem and tuned defaults: the true
-- parameters that generate the data, where descent starts, the input range,
-- the learning rate for GD and for SGD / mini-batch, and the epoch limit.
data Example = Example
  { exKey :: String
  , exModel :: Model
  , exTruth :: Vector Double
  , exStart :: Vector Double
  , exDomain :: (Double, Double)
  , exRate :: Double
  , exStochasticRate :: Double
  , exEpochs :: Int
  , exBlurb :: String
  }

-- | Polynomial of degree n: a·xⁿ + … + const (linear in its parameters).
polynomial :: Int -> Model
polynomial n = linearModel name names ["x"] renderPoly phi
  where
    name = case n of
      1 -> "linear"
      2 -> "quadratic"
      3 -> "cubic"
      _ -> "polynomial of degree " ++ show n
    names
      | n < 26 = [[c] | c <- take (n + 1) ['a' ..]]
      | otherwise = ["w" ++ show k | k <- [n, n - 1 .. 0]]
    renderPoly ps = sumTerms [p ++ power k | (p, k) <- zip ps [n, n - 1 .. 0]]
    power :: Int -> String
    power 0 = ""
    power 1 = "·x"
    power k = "·x" ++ superscript k
    phi :: Scalar a => Vector a -> Vector a
    phi x = fromList [(x ! 0) ^ k | k <- [n, n - 1 .. 0]]

-- | f(x) = a·x + b
linear :: Model
linear = polynomial 1

-- | f(x) = a·x² + b·x + c
quadratic :: Model
quadratic = polynomial 2

-- | f(x) = a·x³ + b·x² + c·x + d
cubic :: Model
cubic = polynomial 3

-- | f(x) = a·e^(b·x)
exponential :: Model
exponential =
  mkModel
    "exponential"
    ["a", "b"]
    ["x"]
    (\ps -> at ps 0 ++ "·e^(" ++ at ps 1 ++ "·x)")
    (\t x -> t ! 0 * exp (t ! 1 * x ! 0))

-- | f(x) = a·sin(b·x + c)
sine :: Model
sine =
  mkModel
    "sine wave"
    ["a", "b", "c"]
    ["x"]
    (\ps -> at ps 0 ++ "·sin(" ++ sumTerms [at ps 1 ++ "·x", at ps 2] ++ ")")
    (\t x -> t ! 0 * sin (t ! 1 * x ! 0 + t ! 2))

-- | f(x) = a / (1 + e^(-b·(x - c)))
logistic :: Model
logistic =
  mkModel
    "logistic curve"
    ["a", "b", "c"]
    ["x"]
    (\ps -> at ps 0 ++ " / (1 + e^(" ++ negateTerm (at ps 1) ++ "·(" ++ sumTerms ["x", negateTerm (at ps 2)] ++ ")))")
    (\t x -> t ! 0 / (1 + exp (negate (t ! 1) * (x ! 0 - t ! 2))))

-- | f(x, y) = a·x + b·y + c, a model with two inputs.
plane :: Model
plane =
  linearModel
    "plane"
    ["a", "b", "c"]
    ["x", "y"]
    (\ps -> sumTerms [at ps 0 ++ "·x", at ps 1 ++ "·y", at ps 2])
    (\x -> fromList [x ! 0, x ! 1, 1])

-- | Safe indexing into rendered parameter strings.
at :: [String] -> Int -> String
at ps i = case drop i ps of
  (p : _) -> p
  [] -> "?"

-- | All built-in examples, as shown by --list.
examples :: [Example]
examples =
  [ Example "linear" linear (fromList [2.5, -1]) (zeros 2) (-2, 2) 0.1 0.05 100 "straight line, the classic first example"
  , Example "quadratic" quadratic (fromList [1.5, -2, 0.5]) (zeros 3) (-2, 2) 0.1 0.02 300 "parabola, still linear in its parameters"
  , Example "cubic" cubic (fromList [0.5, -1, -2, 1]) (zeros 4) (-2, 2) 0.04 0.008 1500 "cubic polynomial, badly conditioned for plain GD"
  , polyExample 4
  , Example "exp" exponential (fromList [1.5, 0.8]) (fromList [1, 0]) (-1, 2) 0.015 0.01 2000 "exponential growth, nonlinear in b"
  , Example "sine" sine (fromList [2, 1.5, 0.5]) (fromList [1, 1.2, 0]) (-3, 3) 0.1 0.02 300 "sine wave, non-convex loss surface"
  , Example "logistic" logistic (fromList [3, 2, 0.5]) (fromList [1, 1, 0]) (-3, 3) 0.5 0.05 1500 "S-shaped curve, nonlinear in all parameters"
  , Example "plane" plane (fromList [1.5, -2, 0.7]) (zeros 3) (-2, 2) 0.1 0.05 100 "two inputs, a flat surface over the (x, y) plane"
  ]

-- | Example problem for a polynomial of degree n.
polyExample :: Int -> Example
polyExample n =
  Example
    ("poly:" ++ show n)
    (polynomial n)
    (fromList (take (n + 1) (cycle [1, -0.5, -2, 1.5, 0.5])))
    (zeros (n + 1))
    (-1, 1)
    (1.5 / fromIntegral (n + 1))
    (0.5 / fromIntegral (n + 1))
    10000
    "polynomial of any degree N (here N = 4)"

-- | Look up an example by name or alias; "poly:N" builds a polynomial of degree N.
findExample :: String -> Either String Example
findExample raw = case map toLower raw of
  name | Just e <- lookup name table -> Right e
  'p' : 'o' : 'l' : 'y' : sep : digits
    | sep `elem` ":=-", not (null digits), all isDigit digits ->
        let n = read digits
         in if n >= 1 && n <= 30
              then Right (polyExample n)
              else Left "polynomial degree must be between 1 and 30"
  _ -> Left ("unknown model '" ++ raw ++ "' (try --list)")
  where
    table = [(exKey e, e) | e <- examples] ++ aliases
    aliases =
      [ ("line", head' "linear")
      , ("exponential", head' "exp")
      , ("sin", head' "sine")
      , ("sigmoid", head' "logistic")
      ]
    head' k = case [e | e <- examples, exKey e == k] of
      (e : _) -> e
      [] -> error ("Model.Catalog: missing example " ++ k)

-- | Default learning rate of an example for the given method.
defaultRate :: Example -> Method -> Double
defaultRate ex FullBatch = exRate ex
defaultRate ex _ = exStochasticRate ex
