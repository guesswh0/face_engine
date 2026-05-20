
## 2026-05-20 - [Numerical stability in vectorized distance calculation]
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup (~6-9x) but can introduce small negative values due to floating-point subtractive cancellation.
**Action:** Always use np.maximum(dists_sq, 0) after calculating squared distances via the expansion formula to ensure numerical stability and avoid NaN when taking square roots.
