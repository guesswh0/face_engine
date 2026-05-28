## 2025-05-22 - Vectorized Distance Calculation in BasicEstimator

**Learning:** Replacing iterative `np.linalg.norm` calls with a vectorized expansion formula (||a-b||² = ||a||² + ||b||² - 2ab) provides a massive speedup (~15x-17x in benchmarks) for nearest-neighbor searches. Pre-calculating fitted norms during `fit` further optimizes the hot path.

**Action:** Always look for O(N) loops over NumPy arrays that can be converted to matrix operations. Ensure numerical stability with `np.maximum(..., 0)` when using the expansion formula. Maintain backward compatibility when adding new pre-calculated state to serialized models.
