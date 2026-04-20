
## 2026-04-20 - [Vectorized BasicEstimator.predict]
**Learning:** Vectorizing the distance calculation using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab yielded an ~4x speedup in the benchmark (and up to 25x in smaller batch scenarios). Pre-calculating fitted norms in 'fit' further reduces computation in 'predict'. Using np.maximum(dists_sq, 0) is essential to avoid negative distances due to floating point errors.
**Action:** Always prefer vectorized operations over iterative np.linalg.norm in hot paths like prediction.
