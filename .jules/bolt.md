## 2025-05-14 - Vectorized Distance Precision
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (~2x) but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` after using the expansion formula to ensure numerical stability and avoid errors in downstream operations (like `np.sqrt` or distance comparisons).
