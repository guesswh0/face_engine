
## 2026-06-01 - Vectorized Euclidean Distance in BasicEstimator
**Learning:** Vectorizing Euclidean distance calculation using the expansion formula ||a-b||² = ||a||² + ||b||² - 2ab provided a ~13x speedup. It's critical to handle potential negative results from floating-point precision issues using np.maximum(dists_sq, 0).
**Action:** Always prefer matrix operations over loops for distance calculations in NumPy; remember to pre-calculate and store norms of the reference set to further optimize.
