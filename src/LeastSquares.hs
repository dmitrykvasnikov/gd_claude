-- | Closed-form least squares via the normal equations, used as a reference solution.
--
-- When the model is linear in its parameters, f(θ, x) = θ · φ(x), stack the
-- feature vectors φ(xᵢ) as the rows of the /design matrix/ X (N × P) and the
-- targets into y. Then the MSE loss is a quadratic bowl:
--
-- >  L(θ) = (1/N) · ‖Xθ - y‖²          ∇L(θ) = (2/N) · Xᵀ(Xθ - y)
--
-- At the bottom of the bowl the gradient is zero, which gives the
-- /normal equations/:
--
-- >  XᵀX · θ = Xᵀy          θ* = (XᵀX)⁻¹ Xᵀy
--
-- This is exactly the point gradient descent walks towards, so comparing the
-- learned θ with θ* shows how close GD/SGD got. Why bother with GD at all
-- then? Solving costs O(P³) and needs the whole data in memory, and it only
-- works for linear models and MSE. GD and SGD work for any differentiable
-- model and loss (try the exp, sine or logistic models).
--
-- The Hessian of L is (2/N)·XᵀX, so the eigenvalues of the XᵀX shown by
-- @gd -x@ also decide the largest stable learning rate (η < N/λ_max(XᵀX)).
module LeastSquares
  ( Normal (..)
  , normalEquations
  ) where

import Dataset
import Math
import Model

-- | Design matrix X, XᵀX, Xᵀy and the solution θ* = (XᵀX)⁻¹Xᵀy.
data Normal = Normal
  { design :: Matrix Double
  , gramMatrix :: Matrix Double
  , moments :: Vector Double
  , solution :: Maybe (Vector Double)
  }

-- | Build and solve the normal equations; Nothing for models nonlinear in θ.
normalEquations :: Model -> Dataset -> Maybe Normal
normalEquations _ [] = Nothing
normalEquations m ds = do
  phi <- features m
  let x = fromRows (map (phi . input) ds)
      xtx = gram x
      xty = transpose x |*^ targets ds
  pure (Normal x xtx xty (solve xtx xty))
