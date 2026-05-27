## 2026-05-27 - Vectorized Euclidean Distance Optimization
**Learning:** Using the expansion formula ||a-b||² = ||a||² + ||b||² - 2ab allows for full vectorization of nearest-neighbor searches in NumPy, providing a ~15x speedup over loop-based distance calculations. However, subtractive cancellation can lead to slightly negative values, so np.maximum(dists_sq, 0) is necessary for stability.
**Action:** Always prefer matrix-based distance calculations for large datasets and include numerical stability guards.
