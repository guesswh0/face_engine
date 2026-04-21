
import time
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def benchmark():
    n_fitted = 1000
    n_predict = 100
    dim = 128

    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)

    # Warmup
    estimator.predict(predict_embeddings[:1])

    start_time = time.time()
    for _ in range(10):
        scores, names = estimator.predict(predict_embeddings)
    end_time = time.time()

    avg_time = (end_time - start_time) / 10
    print(f"Average time for predicting {n_predict} embeddings with {n_fitted} fitted: {avg_time:.6f}s")

if __name__ == "__main__":
    benchmark()
