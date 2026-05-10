## 2025-05-10 - [Numerical Stability in Distance Expansion]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` after the expansion formula and allow slightly relaxed test tolerances (e.g., `atol=1e-5`) if comparing against iterative `np.linalg.norm`.
