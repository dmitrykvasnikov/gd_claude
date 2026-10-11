-- | Generic text layout: colors, number formats, matrices, tables, sparklines, plots.
module Pretty
  ( Color (..)
  , Align (..)
  , paint
  , paints
  , visibleLength
  , padL
  , padR
  , indent
  , fixed
  , sci
  , num
  , short
  , compact
  , vectorInline
  , namedVector
  , bracket
  , matrixLines
  , columnLines
  , hcat
  , table
  , keyValues
  , banner
  , sparkline
  , plot
  ) where

import Data.List (intercalate)
import qualified Data.List as L
import Math
import Numeric (showEFloat, showFFloat)

-- | Terminal colors and styles.
data Color = Bold | Dim | Red | Green | Yellow | Blue | Magenta | Cyan

-- | Column alignment in tables.
data Align = AlignLeft | AlignRight

-- | ANSI SGR code of a color.
code :: Color -> String
code c = case c of
  Bold -> "1"
  Dim -> "2"
  Red -> "31"
  Green -> "32"
  Yellow -> "33"
  Blue -> "34"
  Magenta -> "35"
  Cyan -> "36"

-- | Wrap text in several ANSI styles (no-op when colors are off).
paints :: Bool -> [Color] -> String -> String
paints False _ s = s
paints True [] s = s
paints True cs s = "\ESC[" ++ intercalate ";" (map code cs) ++ "m" ++ s ++ "\ESC[0m"

-- | Wrap text in one ANSI style (no-op when colors are off).
paint :: Bool -> Color -> String -> String
paint on c = paints on [c]

-- | Length of a string as displayed, ignoring ANSI escape sequences.
visibleLength :: String -> Int
visibleLength ('\ESC' : '[' : rest) = visibleLength (drop 1 (dropWhile (/= 'm') rest))
visibleLength (_ : rest) = 1 + visibleLength rest
visibleLength [] = 0

-- | Pad on the left (right-align) to a display width.
padL :: Int -> String -> String
padL n s = replicate (n - visibleLength s) ' ' ++ s

-- | Pad on the right (left-align) to a display width.
padR :: Int -> String -> String
padR n s = s ++ replicate (n - visibleLength s) ' '

-- | Indent every line by n spaces.
indent :: Int -> [String] -> [String]
indent n = map (replicate n ' ' ++)

-- | Display form of NaN and infinities.
special :: Double -> Maybe String
special x
  | isNaN x = Just "NaN"
  | isInfinite x = Just (if x > 0 then "+∞" else "-∞")
  | otherwise = Nothing

-- | Fixed-point format with d decimals (no negative zero).
fixed :: Int -> Double -> String
fixed d x = case special x of
  Just s -> s
  Nothing -> showFFloat (Just d) (if x == 0 then 0 else x) ""

-- | Scientific format with d decimals and a two-digit exponent, e.g. 1.23e-04.
sci :: Int -> Double -> String
sci d x = case special x of
  Just s -> s
  Nothing -> case break (== 'e') (showEFloat (Just d) x "") of
    (mantissa, 'e' : ex) ->
      let e = read ex :: Int
          digits = show (abs e)
       in mantissa ++ "e" ++ (if e < 0 then "-" else "+") ++ replicate (2 - length digits) '0' ++ digits
    (mantissa, _) -> mantissa

-- | Fixed format, switching to scientific for very large or tiny values.
adaptive :: Int -> Double -> Double -> String
adaptive d big x
  | x /= 0 && (abs x >= big || abs x < 10 ^^ negate d) = sci (d - 2) x
  | otherwise = fixed d x

-- | Number format for tables (6 decimals).
num :: Double -> String
num = adaptive 6 1e5

-- | Number format for formulas (4 decimals).
short :: Double -> String
short = adaptive 4 1e6

-- | Shortest readable form: trailing zeros removed.
compact :: Double -> String
compact x
  | x /= 0 && (abs x >= 1e6 || abs x < 1e-4) = short x
  | otherwise = case break (== '.') (fixed 4 x) of
      (whole, '.' : frac) -> case reverse (dropWhile (== '0') (reverse frac)) of
        "" -> whole
        digits -> whole ++ "." ++ digits
      (whole, _) -> whole

-- | Vector on one line, e.g. (1.0, 2.0).
vectorInline :: (Double -> String) -> Vector Double -> String
vectorInline fmt v = "(" ++ intercalate ", " (map fmt (toList v)) ++ ")"

-- | Parameters as "a = 1   b = 2".
namedVector :: [String] -> Vector Double -> String
namedVector names v = intercalate "   " (zipWith (\n x -> n ++ " = " ++ compact x) names (toList v))

-- | Lay out a grid of cells between matrix brackets, columns right-aligned.
bracket :: [[String]] -> [String]
bracket [] = ["( )"]
bracket rows = [top] ++ map line rows ++ [bottom]
  where
    widths = map (maximum . (0 :) . map visibleLength) (L.transpose rows)
    line r = "│ " ++ unwords (zipWith padL widths r) ++ " │"
    inner = sum widths + length widths + 1
    top = "┌" ++ replicate inner ' ' ++ "┐"
    bottom = "└" ++ replicate inner ' ' ++ "┘"

-- | Pretty-print a matrix.
matrixLines :: Matrix Double -> [String]
matrixLines = bracket . map (map num) . toLists

-- | Pretty-print a vector as a column.
columnLines :: Vector Double -> [String]
columnLines = bracket . map (\x -> [num x]) . toList

-- | Place text blocks side by side, vertically centred.
hcat :: [[String]] -> [String]
hcat blocks = foldr (zipWith (++)) (replicate height "") (map fit blocks)
  where
    height = maximum (0 : map length blocks)
    fit b =
      let w = maximum (0 : map visibleLength b)
          above = (height - length b) `div` 2
          below = height - length b - above
       in map (padR w) (replicate above "" ++ b ++ replicate below "")

-- | Box-drawn table with a bold header row.
table :: Bool -> [Align] -> [String] -> [[String]] -> [String]
table c aligns headers rows =
  [ border "╭" "┬" "╮"
  , line (map (paint c Bold) headers)
  , border "├" "┼" "┤"
  ]
    ++ map line rows
    ++ [border "╰" "┴" "╯"]
  where
    widths = map (maximum . (0 :) . map visibleLength) (L.transpose (headers : rows))
    border l m r = paint c Dim (l ++ intercalate m [replicate (w + 2) '─' | w <- widths] ++ r)
    bar = paint c Dim "│"
    line cells = bar ++ intercalate bar [" " ++ align a w x ++ " " | (a, w, x) <- zip3 (aligns ++ repeat AlignRight) widths cells] ++ bar
    align AlignLeft = padR
    align AlignRight = padL

-- | Aligned "key  value" lines.
keyValues :: Bool -> [(String, String)] -> [String]
keyValues c kvs = ["  " ++ paint c Dim (padR w k) ++ "  " ++ v | (k, v) <- kvs]
  where
    w = maximum (0 : map (visibleLength . fst) kvs)

-- | Section heading line.
banner :: Bool -> String -> String
banner c title = paints c [Bold, Cyan] ("━━ " ++ title ++ " " ++ replicate (max 3 (74 - visibleLength title)) '━')

-- | One-line chart of a positive series on a log scale (for loss curves).
sparkline :: Int -> [Double] -> String
sparkline _ [] = ""
sparkline width xs = map glyph sampled
  where
    n = length xs
    w = max 1 (min width n)
    sampled = [xs !! (i * (n - 1) `div` max 1 (w - 1)) | i <- [0 .. w - 1]]
    finite v = not (isNaN v || isInfinite v)
    logs = [logBase 10 (max 1e-300 v) | v <- sampled, finite v]
    (lo, hi) = if null logs then (0, 0) else (minimum logs, maximum logs)
    blocks = "▁▂▃▄▅▆▇█"
    glyph v
      | not (finite v) = '!'
      | hi - lo < 1e-12 = '▄'
      | otherwise =
          let t = (logBase 10 (max 1e-300 v) - lo) / (hi - lo)
           in blocks !! max 0 (min 7 (round (t * 7)))

-- | ASCII scatter plot of data points with a fitted curve drawn over them.
plot :: Bool -> Int -> Int -> [(Double, Double)] -> (Double -> Double) -> [String]
plot _ _ _ [] _ = []
plot c w h pts f = zipWith rowLine labels grid ++ [axis, xLabels]
  where
    xs = map fst pts
    ys = map snd pts
    (xlo, xhi) = widen (minimum xs, maximum xs)
    (dlo, dhi) = widen (minimum ys, maximum ys)
    margin = 0.08 * (dhi - dlo)
    ylo = dlo - margin
    yhi = dhi + margin
    widen (a, b) = if b - a < 1e-12 then (a - 1, b + 1) else (a, b)
    xAt col = xlo + (fromIntegral col + 0.5) / fromIntegral w * (xhi - xlo)
    rowOf :: Double -> Int
    rowOf y = round ((yhi - y) / (yhi - ylo) * fromIntegral (h - 1))
    colOf :: Double -> Int
    colOf x = max 0 (min (w - 1) (floor ((x - xlo) / (xhi - xlo) * fromIntegral w)))
    curveRow col =
      let y = f (xAt col)
       in if isNaN y || isInfinite y || y < ylo || y > yhi then Nothing else Just (rowOf y)
    curve = map curveRow [0 .. w - 1]
    onCurve r col = case (curve !! col, if col > 0 then curve !! (col - 1) else Nothing) of
      (Just a, Just b)
        | b < a -> r > b && r <= a
        | b > a -> r >= a && r < b
      (Just a, _) -> r == a
      _ -> False
    dataCells = [(rowOf y, colOf x) | (x, y) <- pts]
    cell r col = case ((r, col) `elem` dataCells, onCurve r col) of
      (True, True) -> paint c Magenta "◉"
      (True, False) -> paint c Cyan "○"
      (False, True) -> paint c Yellow "•"
      (False, False) -> " "
    grid = [concatMap (cell r) [0 .. w - 1] | r <- [0 .. h - 1]]
    labels = [label r | r <- [0 .. h - 1]]
    label r
      | r == 0 = short yhi
      | r == h - 1 = short ylo
      | r == (h - 1) `div` 2 = short ((yhi + ylo) / 2)
      | otherwise = ""
    lw = maximum (map length [short yhi, short ylo, short ((yhi + ylo) / 2)])
    rowLine l g = paint c Dim (padL lw l ++ (if null l then " │" else " ┤")) ++ g
    axis = paint c Dim (replicate lw ' ' ++ " └" ++ replicate w '─')
    loLabel = short xlo
    hiLabel = short xhi
    xLabels = paint c Dim (replicate (lw + 2) ' ' ++ loLabel ++ replicate (w - length loLabel - length hiLabel) ' ' ++ hiLabel)
