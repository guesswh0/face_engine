## 2025-05-15 - Vectorizing Euclidean Distance with Expansion Formula
**Learning:** Using the expansion formula $||a-b||^2 = ||a||^2 + ||b||^2 - 2ab$ for vectorized distance calculation provides a significant speedup (~7x-17x) compared to per-query loops. However, it can introduce small negative values due to floating-point precision errors (subtractive cancellation), which must be handled with `np.maximum(dists_sq, 0)`.
**Action:** Always use `np.maximum(dists_sq, 0)` when using the expansion formula and allow for small `rtol`/`atol` in tests.
