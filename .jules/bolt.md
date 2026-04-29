## 2025-05-14 - Vectorized Distance Calculation in BasicEstimator
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (~12x) but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation. Using np.maximum(dists_sq, 0) is essential for stability.
**Action:** Always use np.maximum(..., 0) when using the expansion formula for Euclidean distances and pre-calculate norms of static datasets.
