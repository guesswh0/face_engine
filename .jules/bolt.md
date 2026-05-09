## 2025-05-21 - [Vectorized Distance Numerical Precision]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use np.maximum(dists_sq, 0) when using this formula to ensure distances are non-negative.
