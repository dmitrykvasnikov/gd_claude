-- | Umbrella module: everything mathematical (vectors, matrices, automatic differentiation).
module Math
  ( module Math.Vector
  , module Math.Matrix
  , module Math.AutoDiff
  ) where

import Math.AutoDiff
import Math.Matrix
import Math.Vector
