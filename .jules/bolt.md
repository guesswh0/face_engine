## 2025-05-15 - Vectorizing Euclidean Distance Calculation
**Learning:** Using the expansion formula $||a-b||^2 = ||a||^2 + ||b||^2 - 2ab$ for vectorized distance calculation provides a massive speedup (12x in this case) but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` when using the expansion formula to ensure physical correctness and stability for subsequent operations like `np.sqrt` or `np.exp`.
