## 2025-05-15 - [Numerical Precision in Distance Expansion]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (up to 12x) but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` and allow slightly relaxed test tolerances (e.g., `rtol=1e-4`) when verifying against iterative implementations.
