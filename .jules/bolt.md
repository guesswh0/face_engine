## 2025-05-15 - [Vectorized BasicEstimator.predict]
**Learning:** Python loops over NumPy arrays are a massive bottleneck in distance calculations. Vectorizing the `predict` method using the expanded squared Euclidean distance formula (`||a-b||^2 = ||a||^2 + ||b||^2 - 2ab`) provided a ~78x speedup.
**Action:** Always look for iterative distance calculations and replace them with vectorized NumPy operations, especially in model prediction paths.
