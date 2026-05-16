## 2025-05-15 - Vectorized Distance Precision
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (~22x) but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` for numerical stability and allow slightly relaxed tolerances in tests when comparing against standard Euclidean distance implementations.
