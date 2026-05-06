## 2025-05-15 - [Numerical Precision in Vectorized Distance Calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (3x-10x) but can introduce small floating-point discrepancies compared to np.linalg.norm(a-b). This is due to subtractive cancellation.
**Action:** Use `np.maximum(dists_sq, 0)` to avoid negative values from floating point noise and allow slightly relaxed tolerances (e.g., `rtol=1e-4`) in tests comparing vectorized results with original implementations.
