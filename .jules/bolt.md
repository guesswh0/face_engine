## 2025-05-15 - [Numerical Precision in Vectorized Distance Calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (up to 10x) by leveraging BLAS via NumPy. However, it can introduce small negative values due to floating-point precision issues (subtractive cancellation).
**Action:** Always wrap the resulting distance matrix with `np.maximum(dists_sq, 0)` and use slightly relaxed tolerances (e.g., `atol=1e-5`) in unit tests when comparing against standard Euclidean distance.
