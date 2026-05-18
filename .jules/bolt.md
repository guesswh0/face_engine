## 2025-05-15 - Vectorized Distance Calculation with Expansion Formula
**Learning:** Using the expansion formula $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2ab^T$ for distance calculation allows for significant speedup via matrix multiplication (`np.dot`), but it can introduce small negative values due to floating-point inaccuracies.
**Action:** Always use `np.maximum(dists_sq, 0)` when using the expansion formula to ensure numerical stability and avoid issues when taking the square root (if needed) or using it in exponential functions.
