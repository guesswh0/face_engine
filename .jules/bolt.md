## 2025-05-15 - [Vectorizing BasicEstimator.predict]
**Learning:** Iterative distance calculation in Python loops is a major bottleneck for face recognition estimators. Using the squared distance expansion formula (||a-b||^2 = ||a||^2 + ||b||^2 - 2ab) allows for full vectorization of batch distance calculations using NumPy, providing significant speedups (over 2x in this case) especially as the dataset grows.
**Action:** Always prefer vectorized NumPy operations over iterative loops for distance-based calculations in model estimators.
