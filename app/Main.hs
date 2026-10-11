-- | Command line entry point of the gradient descent playground.
module Main (main) where

import Control.Monad (forM, forM_, unless, when)
import Data.List (intercalate)
import Data.Maybe (fromMaybe)
import Dataset
import Descent
import LeastSquares
import Loss
import Math
import Model
import Model.Catalog
import Options
import Pretty
import Report
import Rng
import Runner
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO

-- | Parse arguments and dispatch to help, model list or training.
main :: IO ()
main = do
  hSetEncoding stdout utf8
  hSetEncoding stderr utf8
  args <- getArgs
  case parseOptions args of
    Left err -> failWith (err ++ "Try 'gd --help'.")
    Right o
      | optHelp o -> putStr usage
      | otherwise -> do
          c <- maybe (hIsTerminalDevice stdout) pure (optColor o)
          if optList o then listModels c else run c o

-- | Print an error to stderr and exit with failure.
failWith :: String -> IO a
failWith msg = hPutStrLn stderr ("gd: " ++ msg) >> exitFailure

-- | Print the table of built-in models.
listModels :: Bool -> IO ()
listModels c =
  mapM_ putStrLn $
    [banner c "Available models", ""]
      ++ indent 2 (table c (repeat AlignLeft) ["name", "formula", "description"] rows)
  where
    rows = [[exKey e, formula (exModel e), exBlurb e] | e <- examples]

-- | Prepare data and configs, print the setup, run every requested method and compare them.
run :: Bool -> Options -> IO ()
run c o = do
  ex <- either failWith pure (findExample (optModel o))
  loss <- either failWith pure (lossByName (optLoss o))
  let m = exModel ex
      p = paramCount m
  truth <- traverse (sized "--truth" p) (optTruth o)
  start <- maybe (pure (exStart ex)) (sized "--init" p) (optInit o)
  (ds, groundTruth, source) <- case optData o of
    Just path -> do
      txt <- readFile path
      ds <- either (\e -> failWith (path ++ ": " ++ e)) pure (parseCsv (inputDim m) txt)
      pure (ds, Nothing, path ++ " (" ++ show (length ds) ++ " samples)")
    Nothing -> do
      let theta = fromMaybe (exTruth ex) truth
          range@(lo, hi) = fromMaybe (exDomain ex) (optRange o)
          (ds, _) = synthesize m theta range (optSamples o) (optNoise o) (mkGen (optSeed o))
          domain = intercalate ", " (inputNames m) ++ " ∈ [" ++ compact lo ++ ", " ++ compact hi ++ "]"
      pure (ds, Just theta, show (length ds) ++ " synthetic samples, " ++ domain ++ ", noise σ = " ++ compact (optNoise o) ++ ", seed " ++ show (optSeed o))
  when (null ds) (failWith "the dataset is empty")
  let rateFor meth = fromMaybe (defaultRate ex meth) (optRate o)
      epochs = fromMaybe (exEpochs ex) (optEpochs o)
      every = if optQuiet o then 0 else fromMaybe (max 1 (epochs `div` 20)) (optPrintEvery o)
      display = Display every (optPauseEvery o) c
      normal' = if isSquaredError loss then normalEquations m ds else Nothing
      closed = normal' >>= solution
      withPlot = optPlot o && not (optQuiet o)
      configs =
        [ Config meth (rateFor meth) (optDecay o) (optMomentum o) epochs (optTolerance o) (optSeed o + 1) loss
        | meth <- optMethods o
        ]
  mapM_ putStrLn $
    [banner c "Gradient descent playground", ""]
      ++ keyValues
        c
        ( [ ("model", modelName m ++ "   " ++ paint c Bold (formula m))
          , ("data", source)
          ]
            ++ maybe [] (\t -> [("truth", renderParams short m t)]) groundTruth
            ++ [ ("loss", lossName loss)
               , ("start", namedVector (paramNames m) start)
               , ("rate", rateText rateFor ++ decayText (optDecay o) ++ momentumText (optMomentum o))
               , ("epochs", "at most " ++ show epochs ++ ", stop when loss ≤ " ++ sci 1 (optTolerance o))
               , ("methods", intercalate ", " (map methodTag (optMethods o)))
               ]
        )
  when (optMatrices o) $ do
    mapM_ putStrLn (["", banner c "Dataset", ""] ++ datasetView c m ds)
    forM_ normal' $ \n -> mapM_ putStrLn (["", banner c "Linear algebra view", ""] ++ normalView c m n)
  runs <- forM configs $ \cfg -> do
    mapM_ putStrLn ["", banner c (methodName (method cfg)), ""]
    r <- execute display cfg m ds start
    unless (every == 0 && optPauseEvery o == 0) (putStrLn "")
    mapM_ putStrLn (runSummary c withPlot m ds groundTruth closed r)
    pure r
  when (length runs > 1) $
    mapM_ putStrLn (["", banner c "Comparison", ""] ++ comparison c m groundTruth closed runs)
  where
    sized flag p xs
      | length xs == p = pure (fromList xs)
      | otherwise = failWith (flag ++ ": the model has " ++ show p ++ " parameters, got " ++ show (length xs))
    rateText rateFor =
      intercalate ", " ["η = " ++ compact (rateFor meth) ++ " (" ++ methodTag meth ++ ")" | meth <- optMethods o]
    decayText d = if d > 0 then ", decay " ++ compact d ++ " (η/(1 + d·epoch))" else ""
    momentumText b = if b > 0 then ", momentum β = " ++ compact b else ""
