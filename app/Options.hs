-- | Command line options.
module Options
  ( Options (..)
  , parseOptions
  , usage
  ) where

import Data.Char (isDigit, isSpace, toLower)
import Descent
import System.Console.GetOpt
import Text.Read (readMaybe)

-- | All settings from the command line.
data Options = Options
  { optModel :: String
  , optMethods :: [Method]
  , optRate :: Maybe Double
  , optEpochs :: Maybe Int
  , optDecay :: Double
  , optMomentum :: Double
  , optTolerance :: Double
  , optLoss :: String
  , optSamples :: Int
  , optNoise :: Double
  , optInputNoise :: Double
  , optSeed :: Int
  , optTruth :: Maybe [Double]
  , optInit :: Maybe [Double]
  , optRange :: Maybe (Double, Double)
  , optData :: Maybe FilePath
  , optPrintEvery :: Maybe Int
  , optPauseEvery :: Int
  , optQuiet :: Bool
  , optMatrices :: Bool
  , optPlot :: Bool
  , optColor :: Maybe Bool
  , optList :: Bool
  , optHelp :: Bool
  }

-- | Settings used when an option is not given.
defaults :: Options
defaults =
  Options
    { optModel = "quadratic"
    , optMethods = [FullBatch]
    , optRate = Nothing
    , optEpochs = Nothing
    , optDecay = 0
    , optMomentum = 0
    , optTolerance = 1e-12
    , optLoss = "mse"
    , optSamples = 40
    , optNoise = 0
    , optInputNoise = 0
    , optSeed = 42
    , optTruth = Nothing
    , optInit = Nothing
    , optRange = Nothing
    , optData = Nothing
    , optPrintEvery = Nothing
    , optPauseEvery = 0
    , optQuiet = False
    , optMatrices = False
    , optPlot = True
    , optColor = Nothing
    , optList = False
    , optHelp = False
    }

-- | Option handler: update the settings or report an error.
type Update = Options -> Either String Options

-- | Option descriptions for GetOpt (also used for --help).
options :: [OptDescr Update]
options =
  [ Option "M" ["model"] (ReqArg (\s o -> Right o {optModel = s}) "NAME") "model to fit: linear | quadratic | cubic | poly:N |\nexp | sine | logistic | plane   (default: quadratic)"
  , Option "m" ["method"] (ReqArg (\s o -> (\ms -> o {optMethods = ms}) <$> parseMethods s) "LIST") "comma separated: gd | sgd | mb:N | all\n(default: gd; all = gd,sgd,mb:8)"
  , Option "r" ["rate"] (ReqArg (positive "rate" (\x o -> o {optRate = Just x})) "η") "learning rate (default: chosen per model)"
  , Option "e" ["epochs"] (ReqArg (natural "epochs" (\n o -> o {optEpochs = Just n})) "N") "maximum number of epochs (default: per model)"
  , Option "" ["decay"] (ReqArg (nonNegative "decay" (\x o -> o {optDecay = x})) "D") "learning rate schedule η/(1 + D·epoch) (default: 0)"
  , Option "" ["momentum"] (ReqArg (fraction "momentum" (\x o -> o {optMomentum = x})) "β") "classical momentum, 0 ≤ β < 1 (default: 0)"
  , Option "" ["tol"] (ReqArg (nonNegative "tol" (\x o -> o {optTolerance = x})) "ε") "stop early once loss ≤ ε (default: 1e-12)"
  , Option "l" ["loss"] (ReqArg (\s o -> Right o {optLoss = s}) "NAME") "mse | mae | huber | huber:δ (default: mse)"
  , Option "n" ["samples"] (ReqArg (natural "samples" (\n o -> o {optSamples = n})) "N") "number of synthetic samples (default: 40)"
  , Option "" ["noise"] (ReqArg (nonNegative "noise" (\x o -> o {optNoise = x})) "σ") "gaussian noise added to targets (default: 0)"
  , Option "" ["x-noise"] (ReqArg (nonNegative "x-noise" (\x o -> o {optInputNoise = x})) "σ") "gaussian noise added to inputs, i.e. error in\nmeasuring x (default: 0)"
  , Option "s" ["seed"] (ReqArg (integer "seed" (\n o -> o {optSeed = n})) "N") "random seed for data and shuffling (default: 42)"
  , Option "" ["truth"] (ReqArg (numbers "truth" (\xs o -> o {optTruth = Just xs})) "a,b,..") "true parameters used to generate data"
  , Option "" ["init"] (ReqArg (numbers "init" (\xs o -> o {optInit = Just xs})) "a,b,..") "initial parameters"
  , Option "" ["range"] (ReqArg (\s o -> (\r -> o {optRange = Just r}) <$> parseRange s) "LO:HI") "input domain of synthetic data"
  , Option "d" ["data"] (ReqArg (\s o -> Right o {optData = Just s}) "FILE") "read samples from CSV: x1,..,xk,y per line"
  , Option "p" ["print-every"] (ReqArg (natural0 "print-every" (\n o -> o {optPrintEvery = Just n})) "N") "print every N-th epoch, 0 = none\n(default: about 20 rows per run)"
  , Option "w" ["pause-every"] (ReqArg (natural0 "pause-every" (\n o -> o {optPauseEvery = n})) "N") "wait for a key press every N epochs (default: 0 = never)"
  , Option "q" ["quiet"] (NoArg (\o -> Right o {optQuiet = True})) "no epoch table and no plot, only final results"
  , Option "x" ["matrices"] (NoArg (\o -> Right o {optMatrices = True})) "show dataset, design matrix and normal equations"
  , Option "" ["no-plot"] (NoArg (\o -> Right o {optPlot = False})) "do not draw the ASCII plot"
  , Option "" ["color"] (NoArg (\o -> Right o {optColor = Just True})) "force colored output"
  , Option "" ["no-color"] (NoArg (\o -> Right o {optColor = Just False})) "disable colored output"
  , Option "" ["list"] (NoArg (\o -> Right o {optList = True})) "list available models"
  , Option "h" ["help"] (NoArg (\o -> Right o {optHelp = True})) "show this help"
  ]

-- | Parse the command line into settings.
parseOptions :: [String] -> Either String Options
parseOptions args = case getOpt Permute options args of
  (updates, [], []) -> foldl (>>=) (Right defaults) updates
  (_, extra, []) -> Left ("unexpected argument: " ++ unwords extra)
  (_, _, errs) -> Left (concat errs)

-- | Help text with examples.
usage :: String
usage =
  usageInfo "Usage: gd [OPTIONS]\n\nFit a model with gradient descent / stochastic gradient descent.\n" options
    ++ unlines
      [ ""
      , "Examples:"
      , "  gd                                  quadratic model, full-batch gradient descent"
      , "  gd -M linear -m gd,sgd              compare GD and SGD on f(x) = a·x + b"
      , "  gd -M quadratic -m sgd -w 10        SGD, wait for a key press every 10 epochs"
      , "  gd -M exp -m all --noise 0.2 -q     only the final results of all methods"
      , "  gd -M linear --x-noise 0.5 -q       measurement error in x flattens the fit"
      , "  gd -M poly:5 -x -p 500              degree 5 polynomial, show matrices"
      , "  gd -M linear -d points.csv          fit data from a CSV file"
      ]

-- | Parse a comma separated method list such as "gd,sgd,mb:8".
parseMethods :: String -> Either String [Method]
parseMethods s = concat <$> traverse one (splitOn ',' (map toLower s))
  where
    one "gd" = Right [FullBatch]
    one "batch" = Right [FullBatch]
    one "sgd" = Right [Stochastic]
    one "all" = Right [FullBatch, Stochastic, MiniBatch 8]
    one ('m' : 'b' : ':' : digits) | Just b <- readMaybe digits, b > 0 = Right [MiniBatch b]
    one other = Left ("unknown method '" ++ other ++ "' (expected gd, sgd, mb:N or all)\n")

-- | Parse an input domain "LO:HI".
parseRange :: String -> Either String (Double, Double)
parseRange s = case splitOn ':' s of
  [a, b] | Just lo <- readNumber a, Just hi <- readNumber b, lo < hi -> Right (lo, hi)
  _ -> Left ("invalid range '" ++ s ++ "', expected LO:HI with LO < HI\n")

-- | Split a string on a separator character.
splitOn :: Char -> String -> [String]
splitOn sep s = case break (== sep) s of
  (chunk, []) -> [chunk]
  (chunk, _ : rest) -> chunk : splitOn sep rest

-- | Generic numeric option parser with a validity check.
numberWith :: Read a => String -> (a -> Bool) -> String -> (a -> Options -> Options) -> String -> Update
numberWith name ok expectation set s o = case readNumber s of
  Just x | ok x -> Right (set x o)
  _ -> Left ("--" ++ name ++ ": expected " ++ expectation ++ ", got '" ++ s ++ "'\n")

-- | Read a number, also accepting a leading dot such as ".5" or "-.5".
readNumber :: Read a => String -> Maybe a
readNumber = readMaybe . normalize . dropWhile isSpace
  where
    normalize ('.' : r) = "0." ++ r
    normalize ('-' : '.' : r) = "-0." ++ r
    normalize r = r

-- | Option taking a positive number.
positive :: String -> (Double -> Options -> Options) -> String -> Update
positive name = numberWith name (> 0) "a positive number"

-- | Option taking a non-negative number.
nonNegative :: String -> (Double -> Options -> Options) -> String -> Update
nonNegative name = numberWith name (>= 0) "a non-negative number"

-- | Option taking a number in [0, 1).
fraction :: String -> (Double -> Options -> Options) -> String -> Update
fraction name = numberWith name (\x -> x >= 0 && x < 1) "a number in [0, 1)"

-- | Option taking a positive integer.
natural :: String -> (Int -> Options -> Options) -> String -> Update
natural name = numberWith name (> 0) "a positive integer"

-- | Option taking a non-negative integer.
natural0 :: String -> (Int -> Options -> Options) -> String -> Update
natural0 name = numberWith name (>= 0) "a non-negative integer"

-- | Option taking any integer.
integer :: String -> (Int -> Options -> Options) -> String -> Update
integer name set s o
  | all isDigit (dropWhile (== '-') s) = numberWith name (const True) "an integer" set s o
  | otherwise = Left ("--" ++ name ++ ": expected an integer, got '" ++ s ++ "'\n")

-- | Option taking a comma separated list of numbers.
numbers :: String -> ([Double] -> Options -> Options) -> String -> Update
numbers name set s o = case traverse readNumber (splitOn ',' s) of
  Just xs | not (null xs) -> Right (set xs o)
  _ -> Left ("--" ++ name ++ ": expected comma separated numbers, got '" ++ s ++ "'\n")
