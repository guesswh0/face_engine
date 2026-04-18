import time
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def original_predict(self, embeddings):
    # This is a copy of the original iterative implementation
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
    n_fitted = 10000
    embedding_dim = 128
    n_predict = 1000

    fitted_embeddings = np.random.rand(n_fitted, embedding_dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    predict_embeddings = np.random.rand(n_predict, embedding_dim).astype(np.float32)

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    # Benchmark original
    start_time = time.time()
    scores_orig, classes_orig = original_predict(estimator, predict_embeddings)
    end_time = time.time()
    orig_time = end_time - start_time
    print(f"Original iterative prediction took {orig_time:.4f} seconds")

    # Benchmark optimized
    start_time = time.time()
    scores_opt, classes_opt = estimator.predict(predict_embeddings)
    end_time = time.time()
    opt_time = end_time - start_time
    print(f"Optimized vectorized prediction took {opt_time:.4f} seconds")

    print(f"Speedup: {orig_time / opt_time:.2f}x")

    # Verify results are same
    np.testing.assert_allclose(scores_orig, scores_opt, atol=1e-5)
    assert classes_orig == classes_opt

if __name__ == "__main__":
    benchmark()
