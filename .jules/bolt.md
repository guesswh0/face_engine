## 2025-05-22 - [Vectorization of BasicEstimator.predict]
**Learning:** Vectorizing the distance calculation in `BasicEstimator.predict` using the squared distance expansion formula (||a-b||^2 = ||a||^2 + ||b||^2 - 2ab) yielded a massive performance boost (~85x-100x speedup) for batch predictions. This pattern is highly effective in this codebase where multiple face embeddings are often processed at once.
**Action:** Always prefer NumPy vectorization for distance-based calculations over iterative loops, especially when dealing with embedding vectors.
