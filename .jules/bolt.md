## 2026-04-27 - Vectorized distance calculation in BasicEstimator
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides a significant speedup (up to 5x) over iterative np.linalg.norm in Python loops, especially for batch predictions.
**Action:** Always prefer vectorized operations for distance calculations and pre-calculate norms where possible.
