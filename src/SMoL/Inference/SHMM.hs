{-# LANGUAGE RecordWildCards, OverloadedLists #-}
module SMoL.Inference.SHMM where

import Data.List
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.IO.Unsafe

import Control.Monad.State hiding (state)
import SMoL
import SMoL.Matrix.Sampling
import qualified SHMM as SHMM
import SMoL.Matrix.ProbSeqMatrixUtils
import SparseMatrix hiding (trans)

import Control.Monad.Reader

import SMoL.Tags
import SMoL.Tags.Utils
import SMoL.Inference

sample :: MatSeq s -> IO (Vector s)
sample = (fst <$>) . randToIO . sampleSeq vecDist

printSamples :: Show s => Int -> MatSeq s -> IO ()
printSamples n ms = do
    samples <- replicateM n (sample ms)
    mapM_ print samples

printSamples' :: (Eq s, Show s) => Int -> ProbSeq s -> IO ()
printSamples' n ps = do
  let ms = compileSMoL ps
  samples <- replicateM n (sample ms)
  mapM_ print samples

posteriorSHMM :: (Ord s, Show s) => Emissions Double s -> MatSeq s -> Posterior
posteriorSHMM = posterior SHMM.shmmSummedUnsafe

summedPosteriorSHMM :: (Ord s, Show s) => Emissions Double s -> MatSeq s -> Vector Double
summedPosteriorSHMM = infer SHMM.shmmSummedUnsafe

fullPosteriorSHMM :: (Ord s, Show s) => Emissions Double s -> MatSeq s -> SHMM.VecMat
fullPosteriorSHMM = infer SHMM.shmmFullUnsafe

buildQuerySHMM :: (Ord d, Show d) => ProbSeq d -> Query a -> Emissions Double d -> a
buildQuerySHMM = buildQuery SHMM.shmmSummedUnsafe

runQuery :: (Ord d, Show d) => (ProbSeq d, a) -> (a -> Query b) -> Emissions Double d -> b
runQuery (ps, tags) query ems = buildQuerySHMM ps (query tags) ems

type Queries = Reader [(Posterior, TagIxs)]

runQueries :: (Ord d, Show d) => (ProbSeq d, a) -> (a -> Queries b) -> [Emissions Double d] -> b
runQueries (ps, tags) query ems = runReader (query tags) posts
  where posts = map (runQuery (ps, tags) (const ask)) ems

update :: (Ord b, Show b) => (a -> (ProbSeq b, c)) -> (c -> Query a) -> a -> Emissions Double b -> a
update buildModel buildQuery prior emissions = runQuery (model, vars) buildQuery emissions
  where (model, vars) = buildModel prior

updates :: (Ord b, Show b) => (a -> (ProbSeq b, c)) -> (c -> Query a) -> a -> [Emissions Double b] -> a
updates buildModel buildQuery = foldl (update buildModel buildQuery)

minionSymbols :: Int -> [[Char]]
minionSymbols 0 = [[]]
minionSymbols n = [nt:past | past <- minionSymbols (n-1), nt <- "ACGT"]

simulateEmissionsDelta :: (Ord a) => [a] -> Vector a -> Emissions Double a
simulateEmissionsDelta allSym = simulateEmissions allSym toEm
  where toEm nVal ix = V.replicate nVal 0 V.// [(ix, 1)]

simulateEmissionsUniform :: (Ord a) => [a] -> Prob -> Vector a -> Emissions Double a
simulateEmissionsUniform allSym p = simulateEmissions allSym toEm
  where toEm nVal ix = V.replicate nVal ((1-p) / fromIntegral nVal) V.// [(ix, p)]

simulateEmissions :: (Ord a) => [a] -> (Int -> Int -> Vector Prob) -> Vector a -> Emissions Double a
simulateEmissions allSym toEm path = Emissions {
    emissionMat = V.map toEm' path
  , indexMap = index
  }
  where vals = allSym
        nVal = length vals
        index = Map.fromList $ zip vals [0..]
        toEm' a = let (Just ix) = Map.lookup a index
                  in toEm nVal ix


--ex1Model :: (ProbSeq Char, (Tag Bool, Tag Bool))
ex1Model = runTagGen $ do
  (ps1, a) <- eitherOrM 0.4 (symbol 'a') (symbol 'b')
  (ps2, b) <- eitherOrM 0.5 ps1 (symbol 'c')
  let ps = andThen ps2 (symbol 'd')
  return (ps, (a, b))

--ex1Query :: (Tag Bool, Tag Bool)
--         -> Query (Prob, Prob, Prob, Map Bool Prob)
ex1Query (a, b) = do
  bp <- tagDist b
  anb <- condition1 b id $ event1 a not
  yb <- event1 b id
  ya <- event1 a id
  (post, tagIxs) <- ask
  return (ya, yb)

--ex1 :: Emissions Char -> (Prob, Prob, Prob, Map Bool Prob)
ex1 = runQuery ex1Model ex1Query $ toEms ['a','d']

  -- doesn't work, this is the same issue as before - can only see the maximum?
--(ps1, a) <- finiteDistRepeatM [0, 0.1, 0.2, 0.4, 0.2, 0.1] (state 'a')
{-
two strategies for tagging the repeat:
- distRepeat ps s -> distOver [(states (replicate n s), p) | p <- ps]
- distRepeat ps s -> distRepeat ps s terminator
   the terminator is repeated at the end of each loop, but with a different tag
   doesn't this require the same repeating of paths as above though?
     no - the way it's currently structured, the exit edge is distinct for each iteration, otherwise it'd be geo
   in theory, this could be extended to a slightly more capable version with distinct terminators


here's the plan.
  - keep the terminated repeats as a building block
  - default to a hybrid of the above two strategies, where you hold out one of the repeats as the terminator.
  - write this as a smart constructor, adjusting the tag value range to be [1..n], since can't represent 0.
  - remove the tagging capability from vanilla repeats
-}

ex2Model = runTagGen $ do
  let r = eitherOr 0.5 (symbol 'a') (symbols ['a','a'])
  let r' = possibly 0.5 (symbol 'a')
  -- STIPULATION: The terminator state's assignment must not be influenced by the branch.
  -- Can also do initializer
  (ps1, a) <- finiteDistRepeatTermM (symbol 'a') [0.5, 0.5, 0.5, 0.5] r'
  return (ps1, a)
 -- 0.5 * 0.5 vs 0.5 * 0.5 * 0.5, 0.25 vs 0.125, 2/3 vs 1/3. where is 4/5 vs 1/5 coming from?
ex2Query a = do
  --ask
  tagDist a
-- the tagIxs for a are wrong, should be [[0,1,2], [3,4,5,6,7,8]]
-- it's because of the theory that only the terminal should be used. is that theory wrong?
-- could stipulate that the terminator needs to be of constant length, or that it's length should be independant of the path chosen


-- what calculation is this actually doing? is it really bayesian posterior?
ex2 = runQuery ex2Model ex2Query $ toEms2 ['a','a','a']

ex2Full = fullPosteriorSHMM (toEms2 ['a','a','a']) (compileSMoL $ fst ex2Model)
ex2Summed = summedPosteriorSHMM (toEms2 ['a','a']) (compileSMoL $ fst ex2Model)

toEms2 :: Vector Char -> Emissions Double Char
toEms2 s = Emissions {
    emissionMat = V.map toEmission2 s
  , indexMap = ixMap2
  }

toEmission2 :: Char -> Vector Prob
toEmission2 c = let ix = case Map.lookup c ixMap2 of
                      Just ix -> ix
                      Nothing -> error "in building test emissions, ixMap doesn't character in model"
                    n = Map.size ixMap2
                in V.replicate n 0.05 V.// [(ix, 1 - (fromIntegral . pred $ n) * 0.05)]

ixMap2 = Map.fromList $ zip ['a'] [0..]

toEms :: Vector Char -> Emissions Double Char
toEms s = Emissions {
    emissionMat = V.map toEmission s
  , indexMap = ixMap
  }

toEmission :: Char -> Vector Prob
toEmission c = let (Just ix) = Map.lookup c ixMap
               in V.replicate (Map.size ixMap) 0.05 V.// [(ix, 0.85)]

ixMap = Map.fromList $ zip ['a'..'d'] [0..]

{-
issue for the future:
what "event1 a id" is actually asking is the probability of the first branch of a, given that a is reached.
what does it mean to ask for the probability of the first branch of a without knowing a is reached?
-}
