## 2026-05-01 - Vectorizing distance computation in BasicEstimator

**Learning:** Using the squared distance expansion formula $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a \cdot b$ allows for full vectorization of Euclidean distance calculations using optimized BLAS routines (via `np.dot`). This is significantly faster than `np.linalg.norm` in a loop. Pre-calculating $\|b\|^2$ during the `fit` phase further optimizes the `predict` phase. However, subtractive cancellation can lead to tiny negative values for identical vectors, necessitating `np.maximum(dists_sq, 0)` before further processing (like `np.exp`).

**Action:** Always prefer matrix-based distance calculations for batch operations and pre-calculate norms where possible. Use `np.maximum(..., 0)` to guard against floating-point noise in distance expansion formulas.
