-- | Small self-contained pseudo random number generator (SplitMix64).
module Rng
  ( Gen
  , mkGen
  , nextWord
  , uniform
  , uniformR
  , normal
  , draws
  , shuffle
  ) where

import Data.Bits (shiftR, xor)
import Data.List (sortOn)
import Data.Word (Word64)

-- | Generator state.
newtype Gen = Gen Word64

-- | Create a generator from a seed.
mkGen :: Int -> Gen
mkGen s = snd (nextWord (Gen (fromIntegral s)))

-- | Next 64-bit random word and the advanced generator.
nextWord :: Gen -> (Word64, Gen)
nextWord (Gen s) = (mix s', Gen s')
  where
    s' = s + 0x9E3779B97F4A7C15
    mix z0 =
      let z1 = (z0 `xor` (z0 `shiftR` 30)) * 0xBF58476D1CE4E5B9
          z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
       in z2 `xor` (z2 `shiftR` 31)

-- | Uniform double in [0, 1).
uniform :: Gen -> (Double, Gen)
uniform g = (fromIntegral (w `shiftR` 11) / 9007199254740992, g')
  where
    (w, g') = nextWord g

-- | Uniform double in [lo, hi).
uniformR :: (Double, Double) -> Gen -> (Double, Gen)
uniformR (lo, hi) g = (lo + u * (hi - lo), g')
  where
    (u, g') = uniform g

-- | Standard normal sample via the Box–Muller transform.
normal :: Gen -> (Double, Gen)
normal g = (sqrt (-2 * log (1 - u1)) * cos (2 * pi * u2), g2)
  where
    (u1, g1) = uniform g
    (u2, g2) = uniform g1

-- | Draw n values with a generator-threading sampler.
draws :: Int -> (Gen -> (a, Gen)) -> Gen -> ([a], Gen)
draws n f g0
  | n <= 0 = ([], g0)
  | otherwise =
      let (x, g1) = f g0
          (xs, g2) = draws (n - 1) f g1
       in (x : xs, g2)

-- | Random permutation of a list: attach a random key to every element and
-- sort by the keys. SGD calls this at the start of every epoch, so each
-- sample is visited once per epoch, in a fresh random order.
shuffle :: [a] -> Gen -> ([a], Gen)
shuffle xs g = (map snd (sortOn fst (zip keys xs)), g')
  where
    (keys, g') = draws (length xs) nextWord g
