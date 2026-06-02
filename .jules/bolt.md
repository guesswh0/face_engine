## 2025-05-15 - Vectorizing Euclidean Distance Calculation
**Learning:** Using the expansion formula $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \langle a, b \rangle$ for vectorized distance calculation provides a significant speedup (2.4x in this case) compared to a Python loop. However, it can introduce small negative values due to floating-point precision issues, especially when the distance should be zero.
**Action:** Always use `np.maximum(dists_sq, 0)` when using the expansion formula and allow for small floating-point tolerances in tests.
