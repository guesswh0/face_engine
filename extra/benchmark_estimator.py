import time
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def benchmark():
    # Simulate fitted data: 2000 persons with 128-dim embeddings
    n_fitted = 2000
    dim = 128
    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    # Simulate queries: 500 faces to recognize
    n_queries = 500
    query_embeddings = np.random.rand(n_queries, dim).astype(np.float32)

    # Original implementation style (manual loop for reference in explanation)
    def original_predict(fitted, queries):
        scores = []
        for q in queries:
            distances = np.linalg.norm(fitted - q, axis=1)
            index = np.argmin(distances)
            score = np.exp(-0.5 * distances[index] ** 2)
            scores.append(score)
        return scores

    # Warm up original
    original_predict(fitted_embeddings, query_embeddings[:10])

    start_orig = time.time()
    original_predict(fitted_embeddings, query_embeddings)
    end_orig = time.time()
    orig_time = end_orig - start_orig
    print(f"Original-style prediction time: {orig_time:.4f}s")

    # New implementation
    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    # Warm up new
    estimator.predict(query_embeddings[:10])

    start_new = time.time()
    scores, names = estimator.predict(query_embeddings)
    end_new = time.time()
    new_time = end_new - start_new
    print(f"Vectorized prediction time: {new_time:.4f}s")

    print(f"Speedup: {orig_time / new_time:.2f}x")

if __name__ == "__main__":
    benchmark()
