-- | Training samples: synthetic generation, CSV loading and batching helpers.
module Dataset
  ( Sample (..)
  , Dataset
  , synthesize
  , parseCsv
  , chunksOf
  , targets
  , inputRange
  ) where

import Data.Char (isSpace)
import Data.List (sortOn)
import Math
import Model
import Rng
import Text.Read (readMaybe)

-- | One training example: input vector and target value.
data Sample = Sample
  { input :: Vector Double
  , target :: Double
  }
  deriving (Eq, Show)

-- | A dataset is a list of samples.
type Dataset = [Sample]

-- | Generate n samples y = f(θ, x) + σ·ε with x uniform in the range and ε
-- standard normal noise (the samples are sorted by x for nicer printing).
-- With σ = 0 the true θ fits perfectly and the loss can reach 0. With noise
-- no θ fits exactly: the best achievable loss is about σ², and the least
-- squares optimum moves slightly away from the true θ.
synthesize :: Model -> Vector Double -> (Double, Double) -> Int -> Double -> Gen -> (Dataset, Gen)
synthesize m theta range n sigma g0 = (sortOn (toList . input) samples, g1)
  where
    (samples, g1) = draws n sample g0
    sample g =
      let (xs, g') = draws (inputDim m) (uniformR range) g
          (e, g'') = normal g'
          x = fromList xs
       in (Sample x (predict m theta x + sigma * e), g'')

-- | Parse "x1,…,xk,y" rows; skips blank lines, '#' comments and a header row.
parseCsv :: Int -> String -> Either String Dataset
parseCsv k txt = case numbered of
  [] -> Left "no data rows found"
  (first : rest)
    | Nothing <- traverse parseNumber (fields (snd first)) -> traverse row rest
    | otherwise -> traverse row numbered
  where
    numbered = [(i, l) | (i, l) <- zip [1 :: Int ..] (lines txt), meaningful l]
    meaningful l = case dropWhile isSpace l of
      "" -> False
      ('#' : _) -> False
      _ -> True
    row (i, l) = case traverse parseNumber (fields l) of
      Nothing -> Left ("line " ++ show i ++ ": cannot parse numbers in '" ++ l ++ "'")
      Just xs
        | length xs /= k + 1 ->
            Left ("line " ++ show i ++ ": expected " ++ show (k + 1) ++ " columns (" ++ show k ++ " inputs + target), got " ++ show (length xs))
        | otherwise -> Right (Sample (fromList (init xs)) (last xs))

-- | Split a line on commas, semicolons or tabs and trim the fields.
fields :: String -> [String]
fields s = case break (\c -> c == ',' || c == ';' || c == '\t') s of
  (f, []) -> [trim f]
  (f, _ : rest) -> trim f : fields rest
  where
    trim = dropWhile isSpace . reverse . dropWhile isSpace . reverse

-- | Read a number, accepting forms like ".5", "-.5", "+1" and "2.".
parseNumber :: String -> Maybe Double
parseNumber s = readMaybe (fixup s)
  where
    fixup ('-' : '.' : r) = "-0." ++ r
    fixup ('+' : r) = fixup r
    fixup ('.' : r) = "0." ++ r
    fixup r
      | not (null r) && last r == '.' = r ++ "0"
      | otherwise = r

-- | Split a list into chunks of the given size (the last may be shorter).
-- This is how an epoch's shuffled data is cut into (mini-)batches.
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (h, t) = splitAt (max 1 n) xs in h : chunksOf n t

-- | Vector of all targets.
targets :: Dataset -> Vector Double
targets = fromList . map target

-- | Smallest and largest input value over all samples.
inputRange :: Dataset -> (Double, Double)
inputRange ds = case concatMap (toList . input) ds of
  [] -> (0, 0)
  xs -> (minimum xs, maximum xs)
