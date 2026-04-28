## 2025-05-15 - [Vectorized Distance Calculation Precision]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (~12x) but can introduce small floating-point discrepancies due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` to prevent negative squared distances and allow slightly relaxed test tolerances (e.g., `rtol=1e-4`) when comparing with iterative results.
