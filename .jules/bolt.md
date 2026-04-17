## 2026-04-17 - [Vectorized Distance Calculation in BasicEstimator]
**Learning:** Replacing iterative Python loops with vectorized NumPy operations using the squared distance expansion formula (||a-b||^2 = ||a||^2 + ||b||^2 - 2ab) provides a massive performance boost for batch processing. This pattern leverages optimized BLAS libraries and avoids the high overhead of Python loops and individual `np.linalg.norm` calls.
**Action:** Always look for iterative distance or similarity calculations in model estimators and replace them with vectorized matrix operations.
