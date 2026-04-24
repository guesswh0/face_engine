import os
import time
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def original_predict(self, embeddings):
    """The original unoptimized predict implementation for comparison."""
    if self.class_names is None:
        return [], []

    scores = []
    class_names = []
    for embedding in embeddings:
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        index = np.argmin(distances)
        score = np.exp(-0.5 * distances[index] ** 2)
        scores.append(score)
        class_names.append(self.class_names[index])
    return scores, class_names

def benchmark():
    n_fitted = 2000
    n_predict = 500
    dim = 128

    print(f"Benchmarking with {n_fitted} fitted embeddings and {n_predict} input embeddings (dim={dim})...")

    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    fitted_classes = [f"person_{i}" for i in range(n_fitted)]

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, fitted_classes)

    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)

    # Baseline (original implementation)
    start = time.time()
    orig_scores, orig_classes = original_predict(estimator, predict_embeddings)
    baseline_time = time.time() - start
    print(f"Baseline (original) time: {baseline_time:.4f}s")

    # Current implementation (before my changes, it's the same as original)
    start = time.time()
    curr_scores, curr_classes = estimator.predict(predict_embeddings)
    current_time = time.time() - start
    print(f"Current implementation time: {current_time:.4f}s")

    # Verify current matches original
    # We use a slightly relaxed tolerance because the expansion formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
    # can have small floating point differences compared to direct subtraction.
    np.testing.assert_allclose(orig_scores, curr_scores, rtol=1e-4, atol=1e-7)
    assert orig_classes == curr_classes
    print("Verification passed: Current matches baseline (within tolerance).")

if __name__ == "__main__":
    benchmark()
