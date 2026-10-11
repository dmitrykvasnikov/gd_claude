-- | Dense vectors over any numeric type, implemented on plain lists.
module Math.Vector
  ( Vector
  , fromList
  , toList
  , dim
  , zeros
  , unit
  , (!)
  , vzipWith
  , (^+^)
  , (^-^)
  , (*^)
  , (^/)
  , negateV
  , dot
  , hadamard
  , normSq
  , norm
  , distance
  , average
  , force
  , allFinite
  ) where

infixl 6 ^+^, ^-^
infixl 7 *^, ^/
infixl 9 !

-- | A vector of components, stored as a list.
newtype Vector a = Vector [a]
  deriving (Eq, Ord, Show, Functor, Foldable, Traversable)

-- | Build a vector from a list of components.
fromList :: [a] -> Vector a
fromList = Vector

-- | List of the vector's components.
toList :: Vector a -> [a]
toList (Vector xs) = xs

-- | Number of components (dimension).
dim :: Vector a -> Int
dim = length . toList

-- | Zero vector of the given dimension.
zeros :: Num a => Int -> Vector a
zeros n = Vector (replicate n 0)

-- | Unit basis vector e_i of dimension n (1 at index i, 0 elsewhere).
unit :: Num a => Int -> Int -> Vector a
unit n i = Vector [if j == i then 1 else 0 | j <- [0 .. n - 1]]

-- | Component at a zero-based index, with a bounds check.
(!) :: Vector a -> Int -> a
Vector xs ! i
  | i >= 0 && i < length xs = xs !! i
  | otherwise = error ("Math.Vector.!: index " ++ show i ++ " out of range for dimension " ++ show (length xs))

-- | Combine two vectors component-wise; fails on dimension mismatch.
vzipWith :: (a -> b -> c) -> Vector a -> Vector b -> Vector c
vzipWith f (Vector xs) (Vector ys)
  | length xs == length ys = Vector (zipWith f xs ys)
  | otherwise = error ("Math.Vector: dimension mismatch (" ++ show (length xs) ++ " vs " ++ show (length ys) ++ ")")

-- | Vector addition.
(^+^) :: Num a => Vector a -> Vector a -> Vector a
(^+^) = vzipWith (+)

-- | Vector subtraction.
(^-^) :: Num a => Vector a -> Vector a -> Vector a
(^-^) = vzipWith (-)

-- | Multiply a vector by a scalar.
(*^) :: Num a => a -> Vector a -> Vector a
k *^ v = fmap (k *) v

-- | Divide a vector by a scalar.
(^/) :: Fractional a => Vector a -> a -> Vector a
v ^/ k = fmap (/ k) v

-- | Negate every component.
negateV :: Num a => Vector a -> Vector a
negateV = fmap negate

-- | Component-wise (Hadamard) product.
hadamard :: Num a => Vector a -> Vector a -> Vector a
hadamard = vzipWith (*)

-- | Dot (inner) product.
dot :: Num a => Vector a -> Vector a -> a
dot u v = foldl' (+) 0 (toList (hadamard u v))

-- | Squared Euclidean norm.
normSq :: Num a => Vector a -> a
normSq v = dot v v

-- | Euclidean (L2) norm.
norm :: Floating a => Vector a -> a
norm = sqrt . normSq

-- | Euclidean distance between two vectors.
distance :: Floating a => Vector a -> Vector a -> a
distance u v = norm (u ^-^ v)

-- | Arithmetic mean of a list (0 for an empty list).
average :: Fractional a => [a] -> a
average [] = 0
average xs = foldl' (+) 0 xs / fromIntegral (length xs)

-- | Fully evaluate all components, avoiding thunk build-up in loops.
force :: Vector a -> Vector a
force v = foldr seq () v `seq` v

-- | True when no component is NaN or infinite.
allFinite :: RealFloat a => Vector a -> Bool
allFinite = all (\x -> not (isNaN x || isInfinite x))
