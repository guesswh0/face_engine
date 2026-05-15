import time
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def benchmark():
    N_FITTED = 2000
    N_PREDICT = 500
    DIM = 128

    fitted_embeddings = np.random.rand(N_FITTED, DIM).astype(np.float32)
    class_names = [f"person_{i}" for i in range(N_FITTED)]

    predict_embeddings = np.random.rand(N_PREDICT, DIM).astype(np.float32)

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    start_time = time.time()
    scores, predicted_classes = estimator.predict(predict_embeddings)
    end_time = time.time()

    print(f"Prediction time for {N_PREDICT} embeddings against {N_FITTED} fitted: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    benchmark()
