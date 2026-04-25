## 2025-05-14 - Vectorized Distance Calculation in BasicEstimator
**Learning:** Replacing iterative Euclidean distance calculation with vectorized matrix operations using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab significantly improves performance (approx 5x speedup for 500 input embeddings and 2000 fitted embeddings). Pre-calculating squared norms of fitted embeddings during `fit` further reduces computation time during `predict`.
**Action:** Use vectorized NumPy operations and pre-calculate invariant norms for any distance-based estimators.
