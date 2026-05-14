## 2025-05-15 - Vectorized Distance Calculations with Expansion Formula

**Learning:** Using the expansion formula `||a-b||^2 = ||a||^2 + ||b||^2 - 2ab` for L2 distance calculation is significantly faster than `np.linalg.norm` in a loop, especially for large batches. However, it can introduce small negative values due to floating-point precision issues when the distance is very close to zero.

**Action:** Always use `np.maximum(dists_sq, 0)` after the expansion formula calculation to ensure stability.

## 2025-05-15 - Metadata Alignment in Vectorized Operations

**Learning:** When vectorizing operations that involve filtering or sorting (like `find_faces` limit), any associated metadata (like confidence scores in `extra`) must be precisely sliced using the same indices to maintain alignment. Simply using `[:limit]` is incorrect if the indices were sorted based on a criterion (like area).

**Action:** Track indices during vectorized sorting/filtering and use them to slice all related metadata containers.
