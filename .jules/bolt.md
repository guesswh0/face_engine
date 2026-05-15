## 2025-05-15 - [Numerical Stability in Vectorized Distance Calculation]
**Learning:** Using the expansion formula $||a-b||^2 = ||a||^2 + ||b||^2 - 2ab$ for vectorized Euclidean distance calculation provides significant speedup (~3x in this case) but can introduce small floating-point discrepancies (and even tiny negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` to guard against negative values and allow for slightly relaxed tolerances (e.g., `rtol=1e-4`) when comparing against the naive iterative implementation.
