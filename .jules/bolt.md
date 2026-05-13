## 2025-05-15 - [Numerical Precision in Vectorized Distance Calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized Euclidean distance is much faster but can produce small negative values due to floating-point precision limits.
**Action:** Always use np.maximum(dists_sq, 0) when calculating squared distances with this formula to ensure numerical stability.
