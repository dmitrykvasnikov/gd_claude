-- | Interactive driver: prints epochs while training and optionally waits for key presses.
--
-- 'Descent.descend' returns a lazy list of epochs. This module walks that list
-- one element at a time. Because the list is lazy, epoch k+1 is only computed
-- when the loop asks for it, so pausing here really pauses the training, and
-- pressing q simply stops consuming the list.
module Runner
  ( Display (..)
  , execute
  ) where

import Control.Exception (bracket)
import Control.Monad (when)
import Dataset
import Descent
import GHC.Clock (getMonotonicTime)
import Math
import Model
import Report
import System.IO
import System.IO.Error (catchIOError, isEOFError)

-- | Printing and pausing preferences (0 disables printing/pausing).
data Display = Display
  { printEvery :: Int
  , pauseEvery :: Int
  , colored :: Bool
  }

-- | Loop state while consuming the epoch stream.
data Step = Step
  { pausing :: Bool
  , lastShown :: Maybe Double
  , waited :: Double
  , seen :: [Epoch]
  }

-- | Run training, printing every N-th epoch and pausing every M epochs.
-- Keys while paused: enter/space = next, c = continue without pausing, q = stop.
execute :: Display -> Config -> Model -> Dataset -> Vector Double -> IO Run
execute disp cfg m ds theta0 = do
  when (printEvery disp > 0 || pauseEvery disp > 0) (mapM_ putStrLn (epochHeader c m))
  started <- getMonotonicTime
  (history, quit, idle) <- loop (Step (pauseEvery disp > 0) Nothing 0 []) (descend cfg m ds theta0)
  finished <- getMonotonicTime
  let result = case history of
        [] -> Exhausted
        es@(first : _)
          | quit -> Interrupted
          | otherwise -> outcome cfg (epochLoss first) (last es)
  pure (Run cfg history result (finished - started - idle))
  where
    c = colored disp
    due k n = k > 0 && n `mod` k == 0
    loop st [] = pure (reverse (seen st), False, waited st)
    loop st (e : rest) = do
      let final = null rest
          n = epochNo e
          pauseNow = pausing st && n > 0 && due (pauseEvery disp) n && not final
          shown = pauseNow || (n == 0 && pauseEvery disp > 0) || (printEvery disp > 0 && (n == 0 || final || due (printEvery disp) n))
          shownLoss = if shown then Just (epochLoss e) else lastShown st
          st' = st {lastShown = shownLoss, seen = e : seen st}
      when shown (putStrLn (epochRow c (lastShown st) e))
      if pauseNow
        then do
          t0 <- getMonotonicTime
          key <- askKey (pausePrompt c n)
          t1 <- getMonotonicTime
          let st'' = st' {waited = waited st' + t1 - t0}
          case key of
            'q' -> pure (reverse (seen st''), True, waited st'')
            'c' -> loop st'' {pausing = False} rest
            _ -> loop st'' rest
        else loop st' rest

-- | Show a prompt and read a single key (raw mode on a terminal, EOF counts as 'c').
askKey :: String -> IO Char
askKey prompt = do
  putStr prompt
  hFlush stdout
  tty <- hIsTerminalDevice stdin
  key <-
    if tty
      then bracket raw restore (const getChar)
      else getChar `catchIOError` \e -> if isEOFError e then pure 'c' else ioError e
  putStr (if tty then "\r\ESC[2K" else "\n")
  hFlush stdout
  pure key
  where
    raw = do
      buffering <- hGetBuffering stdin
      echo <- hGetEcho stdin
      hSetBuffering stdin NoBuffering
      hSetEcho stdin False
      pure (buffering, echo)
    restore (buffering, echo) = hSetBuffering stdin buffering >> hSetEcho stdin echo
