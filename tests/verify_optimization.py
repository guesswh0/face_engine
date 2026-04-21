
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def original_predict_logic(fitted_embeddings, input_embeddings, class_names):
    scores = []
    names = []
    for embedding in input_embeddings:
        distances = np.linalg.norm(fitted_embeddings - embedding, axis=1)
        index = np.argmin(distances)
        score = np.exp(-0.5 * distances[index] ** 2)
        scores.append(score)
        names.append(class_names[index])
    return scores, names

def verify():
    n_fitted = 50
    n_predict = 10
    dim = 128

    fitted_embeddings = np.random.rand(n_fitted, dim).astype(np.float32)
    class_names = [f"person_{i}" for i in range(n_fitted)]

    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)

    predict_embeddings = np.random.rand(n_predict, dim).astype(np.float32)

    # Get results from optimized version
    opt_scores, opt_names = estimator.predict(predict_embeddings)

    # Get results from original logic
    orig_scores, orig_names = original_predict_logic(fitted_embeddings, predict_embeddings, class_names)

    # Compare
    names_match = opt_names == orig_names
    scores_match = np.allclose(opt_scores, orig_scores, atol=1e-6)

    print(f"Names match: {names_match}")
    print(f"Scores match: {scores_match}")

    if not names_match or not scores_match:
        print("FAILED: Numerical consistency check failed!")
        exit(1)
    else:
        print("PASSED: Numerical consistency check passed!")

    # Test empty input
    empty_scores, empty_names = estimator.predict(np.array([]).reshape(0, dim))
    if empty_scores == [] and empty_names == []:
        print("PASSED: Empty input check passed!")
    else:
        print("FAILED: Empty input check failed!")
        exit(1)

    # Test backward compatibility (missing _sq_norm_fitted)
    del estimator._sq_norm_fitted
    opt_scores_back, opt_names_back = estimator.predict(predict_embeddings)
    if opt_names_back == orig_names and np.allclose(opt_scores_back, orig_scores, atol=1e-6):
        print("PASSED: Backward compatibility check passed!")
    else:
        print("FAILED: Backward compatibility check failed!")
        exit(1)

if __name__ == "__main__":
    verify()
