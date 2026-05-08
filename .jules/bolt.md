## 2025-05-15 - Vectorized Distance Calculation with Numerical Stability
**Learning:** Using the expansion formula $||a-b||^2 = ||a||^2 + ||b||^2 - 2a \cdot b$ for vectorized distance calculation provides a massive speedup (up to ~10x-12x) over loop-based `np.linalg.norm`. However, it can introduce small negative values due to floating-point precision issues (subtractive cancellation), which must be handled with `np.maximum(dists_sq, 0)` before applying `np.exp` or `np.sqrt`.
**Action:** Always use vectorized matrix operations for similarity/distance calculations in NumPy and include a safety clip for numerical stability.

**Learning:** When optimizing methods that involve model state (like `fit` and `predict`), pre-calculating constant properties (like norms of fitted embeddings) during `fit` saves time in every `predict` call. Using `getattr(self, 'prop', None)` ensures backward compatibility for objects serialized with older versions of the code.
**Action:** Identify and pre-calculate invariant properties in `fit` and use safe attribute access for robustness.
