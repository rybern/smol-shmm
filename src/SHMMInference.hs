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

import Control.Monad.Reader

import Sequence.Tags
import Inference

posteriorSHMM :: (Ord s, Show s) => Emissions s -> MatSeq s -> Posterior
posteriorSHMM = posterior SHMM.shmmSummedUnsafe

buildQuerySHMM :: (Ord d, Show d) => ProbSeq d -> Query a -> Emissions d -> a
buildQuerySHMM = buildQuery SHMM.shmmSummedUnsafe

--example :: Emissions Char -> (Prob, Prob, Prob)
example ems = observe ems $ do
  (ps1, a) <- eitherOrM 0.4 (state 'a') (state 'b')
  (ps2, b) <- eitherOrM 0.5 ps1 (state 'c')
  let ps = andThen ps2 (state 'd')
  return . buildQuerySHMM ps $ do
    anb <- condition1 b id $ event1 a not
    nb <- event1 b not
    ya <- event1 a id
    return (anb, nb, ya)
    --tagDist a
    --(post, ixs) <- ask
    --ad <- tagDist a
    --at <- event1 a id
    --bt <- event1 b id
    --return (post, ixs, ad, at, bt, buildMatSeq ps)

--bug in tagDist
  -- appears to have been fixed by containers 0.8.5.1 -> 0.8.11.0
--bug in tagIxs

test1 = toEms "ad"

toEms :: String -> Emissions Char
toEms s = Emissions {
    emissions = V.map toEmission $ V.fromList s
  , indexMap = ixMap
  }

toEmission :: Char -> Vector Prob
toEmission c = let (Just ix) = Map.lookup c ixMap
               in V.replicate (Map.size ixMap) 0 V.// [(ix, 1)]

ixMap = Map.fromList $ zip ['a'..'d'] [0..]
