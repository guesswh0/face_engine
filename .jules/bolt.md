## 2025-05-15 - Vectorized Distance Calculation in BasicEstimator
**Learning:** Vectorizing Euclidean distance calculation using the expansion formula $||a-b||^2 = ||a||^2 + ||b||^2 - 2ab$ provides a significant speedup (up to 12x in this case) compared to a Python-level loop over `np.linalg.norm`. However, it can introduce tiny negative values due to floating-point precision issues, which must be handled with `np.maximum(dists_sq, 0)`.
**Action:** Always use the expansion formula for bulk distance calculations in NumPy and remember to guard against negative squared distances.

## 2025-05-15 - Backward Compatibility for Persisted Models
**Learning:** When optimizing models that are persisted (e.g., via pickle), adding new pre-calculated attributes (like `fitted_norms_sq`) requires updating the `load` method to ensure existing saved models can still be loaded and will have those attributes correctly initialized.
**Action:** Always check `load`/`__setstate__` methods when adding new state to a class that supports serialization.
