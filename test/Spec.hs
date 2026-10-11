-- | Test suite: linear algebra, autodiff, RNG, parsing, formatting and convergence of GD/SGD.
module Main (main) where

import Control.Monad (unless)
import Data.List (sort)
import Dataset
import Descent
import LeastSquares
import Loss
import Math
import Model
import Model.Catalog
import Pretty
import Rng
import System.Exit (exitFailure)

-- | A named check.
data Test = Test String Bool

-- | Run all tests and exit with failure if any fails.
main :: IO ()
main = do
  results <- mapM report tests
  let failed = length (filter not results)
  putStrLn ""
  putStrLn (show (length results - failed) ++ " passed, " ++ show failed ++ " failed")
  unless (failed == 0) exitFailure

-- | Print a test result and return whether it passed.
report :: Test -> IO Bool
report (Test name ok) = do
  putStrLn ((if ok then "  ✓ " else "  ✗ ") ++ name)
  pure ok

-- | Approximate equality of numbers.
near :: Double -> Double -> Double -> Bool
near eps a b = abs (a - b) <= eps

-- | Approximate equality of vectors.
nearV :: Double -> Vector Double -> Vector Double -> Bool
nearV eps u v = dim u == dim v && and (zipWith (near eps) (toList u) (toList v))

-- | Approximate equality of matrices.
nearM :: Double -> Matrix Double -> Matrix Double -> Bool
nearM eps a b = shape a == shape b && and (zipWith (nearV eps) (toRows a) (toRows b))

-- | Central finite-difference gradient, used to check autodiff.
numericGradient :: (Vector Double -> Double) -> Vector Double -> Vector Double
numericGradient f v = fromList [(f (v ^+^ h i) - f (v ^-^ h i)) / (2 * eps) | i <- [0 .. dim v - 1]]
  where
    eps = 1e-6
    h i = eps *^ unit (dim v) i

-- | Deterministic synthetic dataset for an example.
dataset :: Example -> Double -> Dataset
dataset ex sigma = fst (synthesize (exModel ex) (exTruth ex) (exDomain ex) 40 sigma (mkGen 7))

-- | Example by name (test helper).
example :: String -> Example
example k = either error id (findExample k)

-- | Train an example's model with a method and wrap the result in a Run.
fit :: Method -> String -> Int -> Dataset -> Run
fit meth k epochs ds = Run cfg history (outcome cfg (epochLoss (firstOf history)) (last history)) 0
  where
    ex = example k
    cfg = Config meth (defaultRate ex meth) 0 0 epochs 1e-14 1 mse
    history = descend cfg (exModel ex) ds (exStart ex)

-- | Parameters at the end of a run.
learned :: Run -> Vector Double
learned = maybe (fromList []) epochParams . finalEpoch

-- | All test cases.
tests :: [Test]
tests =
  [ Test "dot product" (dot (fromList [1, 2, 3]) (fromList [4, 5, 6 :: Double]) == 32)
  , Test "vector arithmetic" (2 *^ fromList [1, 2] ^+^ fromList [3, 4] ^-^ fromList [1, 1] == fromList [4, 7 :: Int])
  , Test "norm" (near 1e-12 (norm (fromList [3, 4])) 5)
  , Test "matrix product" (fromLists [[1, 2], [3, 4]] |*| fromLists [[5, 6], [7, 8]] == fromLists [[19, 22], [43, 50 :: Int]])
  , Test "matrix × vector" (fromLists [[1, 2], [3, 4]] |*^ fromList [1, 1] == fromList [3, 7 :: Int])
  , Test "vector × matrix" (fromList [1, 1] ^*| fromLists [[1, 2], [3, 4]] == fromList [4, 6 :: Int])
  , Test "transpose" (transpose (fromLists [[1, 2, 3], [4, 5, 6]]) == fromLists [[1, 4], [2, 5], [3, 6 :: Int]])
  , Test "identity is neutral" (let a = fromLists [[2, 1], [7, 3 :: Int]] in a |*| identity 2 == a && identity 2 |*| a == a)
  , Test "gram matrix" (gram (fromLists [[1, 2], [3, 4]]) == fromLists [[10, 14], [14, 20 :: Int]])
  , Test "outer product" (outer (fromList [1, 2]) (fromList [3, 4]) == fromLists [[3, 4], [6, 8 :: Int]])
  , Test "solve 3×3 system" (maybe False (nearV 1e-10 (fromList [2, 3, -1])) (solve system (fromList [8, -11, -3])))
  , Test "solve needs pivoting" (maybe False (nearV 1e-12 (fromList [1, 2])) (solve (fromLists [[0, 1], [1, 0]]) (fromList [2, 1])))
  , Test "solve rejects singular" (solve (fromLists [[1, 2], [2, 4 :: Double]]) (fromList [1, 2]) == Nothing)
  , Test "solve × matrix round trip" (maybe False (\x -> nearV 1e-10 (system |*^ x) (fromList [1, 2, 3])) (solve system (fromList [1, 2, 3])))
  , Test "matrix addition and scaling" (nearM 1e-12 (madd (mscale 2 system) system) (mscale 3 system))
  , Test "autodiff: polynomial" (nearV 1e-12 (snd (gradient (\v -> (v ! 0) ^ (3 :: Int) * v ! 1) (fromList [2, 5]))) (fromList [60, 8]))
  , Test "autodiff: transcendental" (gradientMatches (\v -> sin (v ! 0) * exp (v ! 1) + log (v ! 0 * v ! 1) / sqrt (v ! 1)) (fromList [0.7, 1.3]))
  , Test "autodiff: division, powers, tanh" (gradientMatches (\v -> (v ! 0 ** v ! 1) / (1 + tanh (v ! 0 - v ! 1))) (fromList [1.4, 0.6]))
  , Test "autodiff value" (near 1e-12 (fst (gradient (\v -> v ! 0 * v ! 1) (fromList [3, 4]))) 12)
  , Test "model parameter gradient = features" (nearV 1e-12 (paramGradient quadratic (fromList [1, 2, 3]) (fromList [2])) (fromList [4, 2, 1]))
  , Test "model parameter gradient (nonlinear)" (lossGradientMatches "logistic")
  , Test "loss gradient matches finite differences" (lossGradientMatches "sine")
  , Test "formula rendering" (formula quadratic == "f(x) = a·x² + b·x + c")
  , Test "fitted rendering handles signs" (renderParams short linear (fromList [2, -1]) == "f(x) = 2.0000·x - 1.0000")
  , Test "shuffle is a permutation" (sort (fst (shuffle [1 .. 100 :: Int] (mkGen 3))) == [1 .. 100])
  , Test "shuffle changes order" (fst (shuffle [1 .. 100 :: Int] (mkGen 3)) /= [1 .. 100])
  , Test "uniform stays in range" (all (\x -> x >= -2 && x < 3) (fst (draws 1000 (uniformR (-2, 3)) (mkGen 9))))
  , Test "normal has mean ≈ 0, variance ≈ 1" normalMoments
  , Test "chunksOf" (chunksOf 3 [1 .. 8 :: Int] == [[1, 2, 3], [4, 5, 6], [7, 8]])
  , Test "csv parsing with header and comments" (fmap length (parseCsv 1 "# comment\nx,y\n1,2\n.5,3\n\n-1,4\n") == Right 3)
  , Test "csv rejects wrong column count" (either (const True) (const False) (parseCsv 2 "1,2\n"))
  , Test "least squares recovers truth" (normalSolution "quadratic" == Just True)
  , Test "GD recovers linear model" (recovers FullBatch "linear" 500 1e-5)
  , Test "SGD recovers linear model" (recovers Stochastic "linear" 200 1e-5)
  , Test "GD recovers quadratic model" (recovers FullBatch "quadratic" 1000 1e-5)
  , Test "SGD recovers quadratic model" (recovers Stochastic "quadratic" 300 1e-5)
  , Test "mini-batch recovers quadratic model" (recovers (MiniBatch 8) "quadratic" 600 1e-5)
  , Test "GD recovers exponential model" (recovers FullBatch "exp" 3000 1e-4)
  , Test "SGD recovers two-input plane" (recovers Stochastic "plane" 200 1e-5)
  , Test "GD with noise matches least squares" noisyMatchesLeastSquares
  , Test "GD loss never increases with a small rate" monotoneLoss
  , Test "huge learning rate is reported as divergence" divergenceDetected
  , Test "momentum converges" momentumConverges
  , Test "pretty matrix lines have equal width" (allSame (map visibleLength (matrixLines system)))
  , Test "table lines have equal width" (allSame (map visibleLength (table True [] ["a", "bb"] [["1", "22222"], ["333", "4"]])))
  , Test "visible length ignores colors" (visibleLength (paint True Red "abc") == 3)
  , Test "scientific formatting" (sci 2 1234.5 == "1.23e+03" && sci 1 0.00012 == "1.2e-04")
  , Test "sparkline length" (length (sparkline 10 [1000, 100, 10, 1, 0.1]) == 5)
  ]
  where
    system = fromLists [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]

-- | Autodiff gradient agrees with finite differences.
gradientMatches :: (forall a. Scalar a => Vector a -> a) -> Vector Double -> Bool
gradientMatches f v = nearV 1e-6 (snd (gradient f v)) (numericGradient f v)

-- | Loss gradient of an example agrees with finite differences.
lossGradientMatches :: String -> Bool
lossGradientMatches k = nearV 1e-5 (snd (lossGradient mse m ds theta)) (numericGradient (lossValue mse m ds) theta)
  where
    ex = example k
    m = exModel ex
    ds = dataset ex 0.1
    theta = exStart ex ^+^ fromList (replicate (paramCount m) 0.3)

-- | Normal equations recover the true parameters of a noise-free example.
normalSolution :: String -> Maybe Bool
normalSolution k = do
  n <- normalEquations (exModel ex) (dataset ex 0)
  s <- solution n
  pure (nearV 1e-9 s (exTruth ex))
  where
    ex = example k

-- | Training recovers the true parameters of a noise-free example.
recovers :: Method -> String -> Int -> Double -> Bool
recovers meth k epochs eps = nearV eps (learned (fit meth k epochs (dataset ex 0))) (exTruth ex)
  where
    ex = example k

-- | On noisy data GD converges to the closed-form least squares solution.
noisyMatchesLeastSquares :: Bool
noisyMatchesLeastSquares = case normalEquations (exModel ex) ds >>= solution of
  Just ls -> nearV 1e-5 (learned (fit FullBatch "quadratic" 2000 ds)) ls
  Nothing -> False
  where
    ex = example "quadratic"
    ds = dataset ex 0.5

-- | With a small rate full-batch GD never increases the loss.
monotoneLoss :: Bool
monotoneLoss = and (zipWith (>=) losses (drop 1 losses))
  where
    losses = map epochLoss (runHistory (fit FullBatch "cubic" 300 (dataset (example "cubic") 0.2)))

-- | A far too large learning rate ends the run as diverged.
divergenceDetected :: Bool
divergenceDetected = outcome cfg (epochLoss (firstOf history)) (last history) == Diverged
  where
    ex = example "linear"
    cfg = Config FullBatch 5 0 0 1000 1e-14 1 mse
    history = descend cfg (exModel ex) (dataset ex 0) (exStart ex)

-- | GD with momentum reaches the true parameters.
momentumConverges :: Bool
momentumConverges = nearV 1e-5 (epochParams (last history)) (exTruth ex)
  where
    ex = example "quadratic"
    cfg = Config FullBatch 0.02 0 0.9 2000 1e-14 1 mse
    history = descend cfg (exModel ex) (dataset ex 0) (exStart ex)

-- | Normal samples have mean ≈ 0 and variance ≈ 1.
normalMoments :: Bool
normalMoments = near 0.05 mu 0 && near 0.05 var 1
  where
    xs = fst (draws 20000 normal (mkGen 11))
    mu = average xs
    var = average [(x - mu) ^ (2 :: Int) | x <- xs]

-- | All elements are equal.
allSame :: Eq a => [a] -> Bool
allSame xs = and (zipWith (==) xs (drop 1 xs))

-- | First epoch of a history.
firstOf :: [Epoch] -> Epoch
firstOf (e : _) = e
firstOf [] = error "no epochs"
