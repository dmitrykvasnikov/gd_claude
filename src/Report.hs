-- | Domain-specific rendering: epoch rows, run summaries, comparisons, matrix views.
module Report
  ( epochHeader
  , epochRow
  , pausePrompt
  , outcomeText
  , runSummary
  , comparison
  , datasetView
  , normalView
  ) where

import Dataset
import Descent
import LeastSquares
import Math
import Model
import Pretty

-- | Arrange the cells of an epoch row into fixed-width columns.
layout :: [String] -> [String] -> String
layout fixedCells paramCells = case fixedCells of
  [e, l, t, g, r] ->
    padL 6 e ++ " │ " ++ padL 10 l ++ " " ++ padR 1 t ++ " │ " ++ padL 8 g ++ " │ " ++ padL 8 r ++ " │" ++ concatMap (padL 14) paramCells
  _ -> unwords (fixedCells ++ paramCells)

-- | Header and rule of the epoch table. How to read the columns:
--
--   * loss: L(θ) on the whole dataset. GD lowers it every epoch (with a sane η).
--     SGD may show ↑ now and then, because each update only sees one sample.
--   * ‖∇L‖: length of the full gradient. It goes to 0 at a minimum. The steps
--     are η·∇L, so they shrink by themselves as training converges.
--   * η: the learning rate of that epoch (changes only with --decay).
--   * parameters: watch them approach the truth / least squares values.
epochHeader :: Bool -> Model -> [String]
epochHeader c m = [paint c Bold line, paint c Dim (map ruleChar line)]
  where
    line = layout ["epoch", "loss", " ", "‖∇L‖", "η"] (paramNames m)
    ruleChar '│' = '┼'
    ruleChar _ = '─'

-- | One epoch as a table row, with a trend arrow against the previous shown loss.
epochRow :: Bool -> Maybe Double -> Epoch -> String
epochRow c previous e =
  layout
    [show (epochNo e), sci 4 l, trend, sci 2 (norm (epochGrad e)), sci 2 (epochRate e)]
    (map num (toList (epochParams e)))
  where
    l = epochLoss e
    trend = case previous of
      Nothing -> " "
      Just p
        | l < p -> paint c Green "↓"
        | l > p -> paint c Red "↑"
        | otherwise -> paint c Dim "="

-- | Prompt shown while waiting for a key press.
pausePrompt :: Bool -> Int -> String
pausePrompt c n =
  paint c Yellow ("  ⏸  paused after epoch " ++ show n)
    ++ paint c Dim "   [enter/space] next · [c] run to the end · [q] stop here "

-- | Colored description of how a run ended.
outcomeText :: Bool -> Config -> Outcome -> String
outcomeText c cfg o = case o of
  Converged -> paint c Green "converged" ++ " (loss ≤ " ++ sci 1 (tolerance cfg) ++ ")"
  Exhausted -> paint c Yellow "finished" ++ " (reached the epoch limit of " ++ show (maxEpochs cfg) ++ ")"
  Diverged -> paint c Red "diverged" ++ " (loss exploded, try a smaller learning rate)"
  Interrupted -> paint c Yellow "stopped by user"

-- | Summary of a run: status, timings, fitted formula, loss curve, parameter table and plot.
runSummary :: Bool -> Bool -> Model -> Dataset -> Maybe (Vector Double) -> Maybe (Vector Double) -> Run -> [String]
runSummary c withPlot m ds truth closed run = case (runHistory run, finalEpoch run) of
  (first : _, Just final) ->
    keyValues
      c
      [ ("status", outcomeText c cfg (runOutcome run))
      , ("epochs", show (epochNo final) ++ "  (" ++ show (epochUpdates final) ++ " parameter updates)")
      , ("time", fixed 3 (runSeconds run) ++ " s")
      , ("final loss", sci 6 (epochLoss final) ++ "   ‖∇L‖ = " ++ sci 3 (norm (epochGrad final)))
      , ("fitted", paint c Bold (renderParams short m (epochParams final)))
      , ("loss curve", lossCurve)
      ]
      ++ [""]
      ++ indent 2 (paramTable c m (epochParams first) (epochParams final) truth closed)
      ++ plotLines final
    where
      losses = map epochLoss (runHistory run)
      lossCurve =
        sci 2 (epochLoss first) ++ " " ++ paint c Green (sparkline 48 losses) ++ " " ++ sci 2 (epochLoss final)
  _ -> ["  (no epochs were run)"]
  where
    cfg = runConfig run
    plotLines final
      | withPlot && inputDim m == 1 =
          [""]
            ++ indent 2 [paint c Dim "data " ++ paint c Cyan "○" ++ paint c Dim "   fitted curve " ++ paint c Yellow "•" ++ paint c Dim "   overlap " ++ paint c Magenta "◉"]
            ++ indent 2 (plot c 64 16 [(input s ! 0, target s) | s <- ds] (\x -> predict m (epochParams final) (fromList [x])))
      | otherwise = []

-- | Table of initial, learned, true and least squares parameters with the error.
paramTable :: Bool -> Model -> Vector Double -> Vector Double -> Maybe (Vector Double) -> Maybe (Vector Double) -> [String]
paramTable c m initial learned truth closed =
  table c (AlignLeft : repeat AlignRight) headers rows
  where
    reference = case (closed, truth) of
      (Just ls, _) -> Just ("|Δ| vs least squares", ls)
      (Nothing, Just t) -> Just ("|Δ| vs truth", t)
      _ -> Nothing
    headers =
      ["param", "initial", "learned"]
        ++ maybe [] (const ["truth"]) truth
        ++ maybe [] (const ["least squares"]) closed
        ++ maybe [] (pure . fst) reference
    rows =
      [ [name, num (initial ! i), paint c Bold (num (learned ! i))]
          ++ maybe [] (\t -> [num (t ! i)]) truth
          ++ maybe [] (\ls -> [num (ls ! i)]) closed
          ++ maybe [] (\(_, r) -> [errorCell (abs (learned ! i - r ! i))]) reference
      | (i, name) <- zip [0 ..] (paramNames m)
      ]
    errorCell d
      | d < 1e-3 = paint c Green (sci 2 d)
      | d < 1e-1 = paint c Yellow (sci 2 d)
      | otherwise = paint c Red (sci 2 d)

-- | Side-by-side comparison of several runs.
comparison :: Bool -> Model -> Maybe (Vector Double) -> Maybe (Vector Double) -> [Run] -> [String]
comparison c m truth closed runs =
  indent 2 (table c (AlignLeft : repeat AlignRight) headers (map row runs ++ extra))
  where
    headers = ["method", "epochs", "updates", "time", "final loss"] ++ paramNames m
    row r = case finalEpoch r of
      Just e ->
        [ methodTag (method (runConfig r))
        , show (epochNo e)
        , show (epochUpdates e)
        , fixed 3 (runSeconds r) ++ " s"
        , sci 4 (epochLoss e)
        ]
          ++ map num (toList (epochParams e))
      Nothing -> [methodTag (method (runConfig r)), "-", "-", "-", "-"] ++ map (const "-") (paramNames m)
    extra =
      maybe [] (\t -> [paint c Dim "truth" : replicate 4 "" ++ map num (toList t)]) truth
        ++ maybe [] (\ls -> [paint c Dim "least squares" : replicate 4 "" ++ map num (toList ls)]) closed

-- | Table with the first rows of the dataset.
datasetView :: Bool -> Model -> Dataset -> [String]
datasetView c m ds =
  indent 2 (table c (repeat AlignRight) ("#" : inputNames m ++ ["target"]) rows)
    ++ [paint c Dim ("  … " ++ show hidden ++ " more rows") | hidden > 0]
  where
    shown = 8
    hidden = length ds - shown
    rows = [show i : map num (toList (input s)) ++ [num (target s)] | (i, s) <- zip [1 :: Int ..] (take shown ds)]

-- | Design matrix, normal equations and their closed-form solution.
normalView :: Bool -> Model -> Normal -> [String]
normalView c m n =
  [ "  " ++ paint c Bold "Design matrix X" ++ paint c Dim (" (" ++ show rows ++ "×" ++ show cols ++ ", one row of features per sample" ++ truncated ++ ")")
  ]
    ++ indent 4 (bracket designCells)
    ++ ["", "  " ++ paint c Bold "Normal equations  XᵀX · θ = Xᵀy"]
    ++ indent 4 (hcat [matrixLines (gramMatrix n), ["  ·  "], bracket (map pure (paramNames m)), ["  =  "], columnLines (moments n)])
    ++ [""]
    ++ [ "  " ++ paint c Bold "Closed-form solution  θ* = (XᵀX)⁻¹ Xᵀy" ++ "   " ++ maybe (paint c Red "XᵀX is singular") (namedVector (paramNames m)) (solution n)
       ]
  where
    (rows, cols) = shape (design n)
    limit = 6
    truncated = if rows > limit then ", first " ++ show limit ++ " shown" else ""
    designCells =
      map (map num) (take limit (toLists (design n)))
        ++ [replicate cols "⋮" | rows > limit]
