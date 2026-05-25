## 2025-05-22 - [Vectorization of distance calculations]
**Learning:** Vectorizing distance calculations using the expansion formula ||a-b||² = ||a||² + ||b||² - 2ab provides a massive speedup (~10x) over looping and using `np.linalg.norm`. It also allows for pre-calculating the norms of the fitted dataset, further reducing computation time per query.
**Action:** Always prefer matrix-based vectorization for batch processing in numerical computations. Ensure backward compatibility when adding new pre-calculated state variables.
