## 2025-05-15 - Vectorized Distance Calculation with Numerical Stability
**Learning:** Using the squared distance expansion formula $||a - b||^2 = ||a||^2 + ||b||^2 - 2ab^T$ for vectorization provides significant speedup (up to 10-12x) but can introduce small negative values due to floating-point precision errors (subtractive cancellation).
**Action:** Always use `np.maximum(dists_sq, 0)` when using the expansion formula and allow for slightly relaxed tolerances (e.g., `rtol=1e-4`) in regression tests comparing against `np.linalg.norm`.
