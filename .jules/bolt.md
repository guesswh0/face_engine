## 2025-05-15 - [Numerical stability in vectorized distance calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup but can introduce small floating-point discrepancies (negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` after the expansion formula to ensure physical validity and use slightly relaxed test tolerances (e.g., `rtol=1e-4`) when comparing with iterative results.
