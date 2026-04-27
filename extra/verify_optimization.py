import numpy as np
import time
import sys
import os

# Add current directory to sys.path to import face_engine
sys.path.append(os.getcwd())

from face_engine.models.basic_estimator import BasicEstimator

def original_predict(estimator, embeddings):
    if estimator.class_names is None:
        raise ValueError("Model is not fitted yet!")

    scores = []
    class_names = []
    for embedding in embeddings:
        distances = np.linalg.norm(estimator.embeddings - embedding, axis=1)
        index = np.argmin(distances)
        score = np.exp(-0.5 * distances[index] ** 2)
        scores.append(score)
        class_names.append(estimator.class_names[index])
    return scores, class_names

def optimized_predict_proposal(estimator, embeddings):
    if estimator.class_names is None:
        raise ValueError("Model is not fitted yet!")

    # Ensure inputs are numpy arrays
    A = np.asanyarray(embeddings)
    B = estimator.embeddings

    # Expansion formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
    # A: (N, D), B: (M, D)

    sq_norm_A = np.sum(A**2, axis=1, keepdims=True) # (N, 1)
    if hasattr(estimator, '_sq_norm_fitted'):
        sq_norm_B = estimator._sq_norm_fitted # (M,)
    else:
        sq_norm_B = np.sum(B**2, axis=1) # (M,)

    # dists_sq = sq_norm_A + sq_norm_B - 2 * A.dot(B.T)
    # Using @ operator for matrix multiplication
    dists_sq = sq_norm_A + sq_norm_B - 2 * (A @ B.T)

    # Numerical stability: distances squared should be >= 0
    dists_sq = np.maximum(dists_sq, 0)

    indices = np.argmin(dists_sq, axis=1)
    min_d2 = dists_sq[np.arange(len(A)), indices]

    scores = np.exp(-0.5 * min_d2).tolist()
    class_names = [estimator.class_names[i] for i in indices]

    return scores, class_names

def verify():
    dim = 128
    n_fitted = 1000
    n_predict = 100

    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)
    # Simulate pre-calculated norm
    estimator._sq_norm_fitted = np.sum(fitted_embeddings**2, axis=1)

    print("Verifying correctness...")
    s1, n1 = original_predict(estimator, predict_embeddings)
    s2, n2 = optimized_predict_proposal(estimator, predict_embeddings)

    np.testing.assert_array_almost_equal(s1, s2, decimal=5)
    assert n1 == n2, "Class names mismatch"
    print("Correctness verified!")

    print("\nBenchmarking...")
    # Larger scale for benchmark
    n_fitted = 2000
    n_predict = 500
    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]
    estimator.fit(fitted_embeddings, class_names)
    estimator._sq_norm_fitted = np.sum(fitted_embeddings**2, axis=1)

    start = time.time()
    original_predict(estimator, predict_embeddings)
    orig_time = time.time() - start
    print(f"Original predict:  {orig_time:.4f}s")

    start = time.time()
    optimized_predict_proposal(estimator, predict_embeddings)
    opt_time = time.time() - start
    print(f"Optimized predict: {opt_time:.4f}s")

    print(f"Speedup: {orig_time / opt_time:.2f}x")

if __name__ == "__main__":
    verify()
