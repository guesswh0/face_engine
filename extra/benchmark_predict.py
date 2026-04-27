import numpy as np
import time
import sys
import os

# Add current directory to sys.path to import face_engine
sys.path.append(os.getcwd())

from face_engine.models.basic_estimator import BasicEstimator

def benchmark():
    dim = 128
    n_fitted = 2000
    n_predict = 500

    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    start = time.time()
    scores, names = estimator.predict(predict_embeddings)
    end = time.time()

    print(f"Time taken for {n_predict} predictions against {n_fitted} fitted: {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark()
