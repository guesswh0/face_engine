## 2025-05-14 - Vectorized Distance Calculation Precision
**Learning:** Using the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab for vectorized distance calculation provides significant speedup but can introduce small floating-point discrepancies (tiny negative values) due to subtractive cancellation.
**Action:** Always use `np.maximum(dists_sq, 0)` when using this expansion to ensure numerical stability, especially when the result is used in functions like `np.exp` or `np.sqrt`.

## 2025-05-14 - Backward Compatibility for Persisted Models
**Learning:** Optimizing model internal state (e.g., pre-calculating norms) can break loading of existing pickled models if the new code expects attributes that weren't present when the model was saved.
**Action:** Use `getattr(self, "attr", None)` with a fallback calculation when accessing newly added pre-calculated attributes in `predict` or `__setstate__` to maintain backward compatibility with older model versions.
