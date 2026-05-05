## 2026-05-05 - [Numerical stability in vectorized distance calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` to ensure non-negative distances and allow for small floating-point noise in comparisons.
