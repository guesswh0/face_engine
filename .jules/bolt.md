## 2025-05-15 - [Vectorizing Batch Distance Calculation]
**Learning:** Replacing iterative `np.linalg.norm` calls in a Python loop with a vectorized implementation using the squared distance expansion formula ($||a-b||^2 = ||a||^2 + ||b||^2 - 2ab$) yielded a ~15x speedup for batch predictions (100 input embeddings, 1000 fitted). Using `np.dot` for the cross-term is the key performance driver. Pre-calculating fitted norms during `fit` further avoids redundant work.
**Action:** Always prefer vectorized NumPy operations for distance-based estimators. Use the squared distance expansion for $O(M \times N)$ distance matrices when memory allows.

## 2025-05-15 - [Numerical Stability in Squared Distance Expansion]
**Learning:** The squared distance expansion formula can occasionally produce tiny negative values (e.g., -1e-15) due to floating-point precision errors, which can cause issues with downstream functions like `np.sqrt` or `np.exp`.
**Action:** Always use `np.maximum(dists_sq, 0)` after calculating the expanded distance matrix to ensure numerical stability.
