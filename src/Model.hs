-- | Abstraction of a parametric model f(θ, x).
--
-- A model is a family of functions of the input x, indexed by parameters θ.
-- Training picks one member of the family: the one that fits the data best.
--
-- >  f(x) = a·x + b             θ = (a, b)       a straight line
-- >  f(x) = a·x² + b·x + c      θ = (a, b, c)    a parabola
-- >  f(x) = a·e^(b·x)           θ = (a, b)       an exponential
--
-- GD only needs two things from a model: its value f(θ, x) and its gradient
-- ∇_θ f(θ, x), i.e. how the output changes when each parameter is nudged.
-- A model is written once, polymorphically in the number type ('Scalar'),
-- so the same definition evaluates on Doubles and differentiates on dual
-- numbers. Nobody writes derivatives by hand.
--
-- Note that the first two models are /linear in their parameters/ even though
-- the parabola is not linear in x: f = θ · φ(x) with features φ(x) = (x², x, 1).
-- For such models the MSE loss is a convex bowl with a single minimum, so GD
-- cannot get stuck. Nonlinear models like a·sin(b·x + c) can have several
-- valleys, and where you start decides which one you end up in.
module Model
  ( Model (..)
  , mkModel
  , linearModel
  , paramCount
  , inputDim
  , isLinear
  , predict
  , paramGradient
  , signature
  , formula
  , renderParams
  , sumTerms
  , negateTerm
  , superscript
  ) where

import Data.List (intercalate)
import Math

-- | A parametric model: names, a pretty printer and the function itself.
-- 'features' is present when the model is linear in its parameters
-- (f = θ · φ(x)), which enables the closed-form least squares solution.
data Model = Model
  { modelName :: String
  , paramNames :: [String]
  , inputNames :: [String]
  , render :: [String] -> String
  , function :: forall a. Scalar a => Vector a -> Vector a -> a
  , features :: Maybe (Vector Double -> Vector Double)
  }

-- | Build a general (possibly nonlinear) model.
mkModel
  :: String
  -> [String]
  -> [String]
  -> ([String] -> String)
  -> (forall a. Scalar a => Vector a -> Vector a -> a)
  -> Model
mkModel name ps xs r f =
  Model
    { modelName = name
    , paramNames = ps
    , inputNames = xs
    , render = r
    , function = f
    , features = Nothing
    }

-- | Build a model that is linear in its parameters from its feature map φ(x).
linearModel
  :: String
  -> [String]
  -> [String]
  -> ([String] -> String)
  -> (forall a. Scalar a => Vector a -> Vector a)
  -> Model
linearModel name ps xs r phi =
  Model
    { modelName = name
    , paramNames = ps
    , inputNames = xs
    , render = r
    , function = \theta x -> dot theta (phi x)
    , features = Just phi
    }

-- | Number of parameters.
paramCount :: Model -> Int
paramCount = length . paramNames

-- | Number of input variables.
inputDim :: Model -> Int
inputDim = length . inputNames

-- | True when the model is linear in its parameters.
isLinear :: Model -> Bool
isLinear = maybe False (const True) . features

-- | Evaluate the model for given parameters and input.
predict :: Model -> Vector Double -> Vector Double -> Double
predict m = function m

-- | Gradient of the model output with respect to the parameters, ∂f/∂θ.
paramGradient :: Model -> Vector Double -> Vector Double -> Vector Double
paramGradient m theta x = snd (gradient (\t -> function m t (fmap constant x)) theta)

-- | Left-hand side such as "f(x, y)".
signature :: Model -> String
signature m = "f(" ++ intercalate ", " (inputNames m) ++ ")"

-- | Symbolic formula using the parameter names.
formula :: Model -> String
formula m = signature m ++ " = " ++ render m (paramNames m)

-- | Formula with numeric parameter values substituted.
renderParams :: (Double -> String) -> Model -> Vector Double -> String
renderParams fmt m theta = signature m ++ " = " ++ render m (map fmt (toList theta))

-- | Join terms with " + ", turning "+ -t" into "- t".
sumTerms :: [String] -> String
sumTerms [] = "0"
sumTerms (t : ts) = t ++ concatMap glue ts
  where
    glue ('-' : rest) = " - " ++ rest
    glue term = " + " ++ term

-- | Textually negate a term.
negateTerm :: String -> String
negateTerm ('-' : rest) = rest
negateTerm term = '-' : term

-- | Render an integer with Unicode superscript digits.
superscript :: Int -> String
superscript = map sup . show
  where
    sup c = case lookup c (zip "0123456789-" "⁰¹²³⁴⁵⁶⁷⁸⁹⁻") of
      Just s -> s
      Nothing -> c
