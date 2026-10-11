-- | Dense row-major matrices and a Gaussian-elimination linear solver.
module Math.Matrix
  ( Matrix
  , fromLists
  , toLists
  , fromRows
  , toRows
  , fromColumns
  , columns
  , rowCount
  , colCount
  , shape
  , identity
  , transpose
  , (|*|)
  , (|*^)
  , (^*|)
  , mscale
  , madd
  , outer
  , gram
  , solve
  ) where

import Data.List (maximumBy)
import qualified Data.List as L
import Data.Ord (comparing)
import Math.Vector

infixl 7 |*|, |*^, ^*|

-- | A matrix stored as a list of rows.
newtype Matrix a = Matrix [[a]]
  deriving (Eq, Show, Functor, Foldable, Traversable)

-- | Build a matrix from rows given as lists; rows must have equal length.
fromLists :: [[a]] -> Matrix a
fromLists rs
  | all ((== width) . length) rs = Matrix rs
  | otherwise = error "Math.Matrix.fromLists: rows have different lengths"
  where
    width = case rs of
      [] -> 0
      (r : _) -> length r

-- | Rows of the matrix as lists.
toLists :: Matrix a -> [[a]]
toLists (Matrix rs) = rs

-- | Build a matrix from row vectors.
fromRows :: [Vector a] -> Matrix a
fromRows = fromLists . map toList

-- | Rows of the matrix as vectors.
toRows :: Matrix a -> [Vector a]
toRows = map fromList . toLists

-- | Build a matrix from column vectors.
fromColumns :: [Vector a] -> Matrix a
fromColumns = transpose . fromRows

-- | Columns of the matrix as vectors.
columns :: Matrix a -> [Vector a]
columns = toRows . transpose

-- | Number of rows.
rowCount :: Matrix a -> Int
rowCount = length . toLists

-- | Number of columns.
colCount :: Matrix a -> Int
colCount m = case toLists m of
  [] -> 0
  (r : _) -> length r

-- | (rows, columns) of the matrix.
shape :: Matrix a -> (Int, Int)
shape m = (rowCount m, colCount m)

-- | n×n identity matrix.
identity :: Num a => Int -> Matrix a
identity n = Matrix [[if i == j then 1 else 0 | j <- [1 .. n]] | i <- [1 .. n]]

-- | Matrix transpose.
transpose :: Matrix a -> Matrix a
transpose (Matrix rs) = Matrix (L.transpose rs)

-- | Matrix-matrix product.
(|*|) :: Num a => Matrix a -> Matrix a -> Matrix a
a |*| b
  | colCount a == rowCount b = Matrix [[dot r c | c <- cs] | r <- toRows a]
  | otherwise = error ("Math.Matrix.|*|: cannot multiply " ++ showShape a ++ " by " ++ showShape b)
  where
    cs = columns b

-- | Matrix-vector product A·v.
(|*^) :: Num a => Matrix a -> Vector a -> Vector a
m |*^ v
  | colCount m == dim v = fromList [dot r v | r <- toRows m]
  | otherwise = error ("Math.Matrix.|*^: cannot multiply " ++ showShape m ++ " by vector of dimension " ++ show (dim v))

-- | Vector-matrix product vᵀ·A.
(^*|) :: Num a => Vector a -> Matrix a -> Vector a
v ^*| m = transpose m |*^ v

-- | Multiply every entry by a scalar.
mscale :: Num a => a -> Matrix a -> Matrix a
mscale k = fmap (k *)

-- | Entry-wise matrix addition.
madd :: Num a => Matrix a -> Matrix a -> Matrix a
madd a b
  | shape a == shape b = Matrix (zipWith (zipWith (+)) (toLists a) (toLists b))
  | otherwise = error ("Math.Matrix.madd: shape mismatch " ++ showShape a ++ " vs " ++ showShape b)

-- | Outer product u·vᵀ.
outer :: Num a => Vector a -> Vector a -> Matrix a
outer u v = Matrix [[x * y | y <- toList v] | x <- toList u]

-- | Gram matrix AᵀA.
gram :: Num a => Matrix a -> Matrix a
gram m = transpose m |*| m

-- | Solve A·x = b for square A; Nothing when A is singular or shapes disagree.
-- Gaussian elimination: use row operations to turn A into an upper triangular
-- matrix (the last unknown then stands alone in the last row), then solve
-- from the bottom up. Used for the closed-form least squares reference.
solve :: (Ord a, Fractional a) => Matrix a -> Vector a -> Maybe (Vector a)
solve a b
  | rows /= cols || rows /= dim b = Nothing
  | otherwise = fromList . backSubstitute <$> eliminate (zipWith (\r y -> r ++ [y]) (toLists a) (toList b))
  where
    (rows, cols) = shape a

-- | Forward elimination with partial pivoting on augmented rows, giving an upper-triangular system.
eliminate :: (Ord a, Fractional a) => [[a]] -> Maybe [[a]]
eliminate [] = Just []
eliminate rows = do
  (pivot, rest) <- pickPivot rows
  case pivot of
    (p : ps) | abs p > 1e-12 -> (pivot :) <$> eliminate (map (reduce p ps) rest)
    _ -> Nothing
  where
    reduce p ps (x : xs) = zipWith (\u w -> u - (x / p) * w) xs ps
    reduce _ _ [] = []

-- | Pick the row with the largest leading coefficient (in absolute value) and return it with the remaining rows.
pickPivot :: (Ord a, Num a) => [[a]] -> Maybe ([a], [[a]])
pickPivot [] = Nothing
pickPivot rows = Just (best, [r | (i, r) <- indexed, i /= k])
  where
    indexed = zip [0 :: Int ..] rows
    (k, best) = maximumBy (comparing (leading . snd)) indexed
    leading (x : _) = abs x
    leading [] = 0

-- | Back substitution on an upper-triangular augmented system.
backSubstitute :: Fractional a => [[a]] -> [a]
backSubstitute = foldr step []
  where
    step (p : rest@(_ : _)) xs = (last rest - sum (zipWith (*) (init rest) xs)) / p : xs
    step _ xs = xs

-- | Render a matrix shape as "r×c" for error messages.
showShape :: Matrix a -> String
showShape m = show (rowCount m) ++ "×" ++ show (colCount m)
