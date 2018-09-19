{-# LANGUAGE RecordWildCards #-}
module SHMMInference where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.IO.Unsafe

import Sequence
import qualified SHMM as SHMM
import Sequence.Matrix.ProbSeqMatrixUtils
import SparseMatrix hiding (trans)

import Sequence.Tags
import Inference

posteriorSHMM :: (Ord s, Show s) => Emissions s -> MatSeq s -> Posterior
posteriorSHMM = posterior SHMM.shmmSummedUnsafe

buildQuerySHMM :: (Ord d, Show d) => ProbSeq d -> Query a -> Emissions d -> a
buildQuerySHMM = buildQuery SHMM.shmmSummedUnsafe

example :: Emissions Char -> (Prob, Prob)
example ems = observe ems $ do
  (ps1, a) <- eitherOr' 0.4 (state 'a') (state 'b')
  (ps2, b) <- eitherOr' 0.5 ps1 (state 'c')
  let ps = andThen ps2 (state 'd')
  return . buildQuerySHMM ps $ do
    anb <- condition1 b id $ event1 a not
    na <- event1 b not
    return (anb, na)
