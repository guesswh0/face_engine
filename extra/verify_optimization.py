import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

def verify():
    N = 100
    M = 10
    D = 128

    fitted_embeddings = np.random.rand(N, D).astype(np.float32)
    input_embeddings = np.random.rand(M, D).astype(np.float32)
    class_names = [f"person_{i}" for i in range(N)]

    # Original implementation logic
    def original_predict(embeddings, fitted_embeddings, class_names):
        scores = []
        preds = []
        for embedding in embeddings:
            distances = np.linalg.norm(fitted_embeddings - embedding, axis=1)
            index = np.argmin(distances)
            score = np.exp(-0.5 * distances[index] ** 2)
            scores.append(score)
            preds.append(class_names[index])
        return scores, preds

    # New implementation
    estimator = BasicEstimator()
    estimator.fit(fitted_embeddings, class_names)
    new_scores, new_preds = estimator.predict(input_embeddings)

    orig_scores, orig_preds = original_predict(input_embeddings, fitted_embeddings, class_names)

    assert new_preds == orig_preds, f"Predictions mismatch: {new_preds} != {orig_preds}"
    np.testing.assert_allclose(new_scores, orig_scores, rtol=1e-5, err_msg="Scores mismatch")
    print("Verification successful: Vectorized implementation matches original logic.")

    # Test backward compatibility (without norms_sq)
    del estimator.norms_sq
    new_scores_compat, new_preds_compat = estimator.predict(input_embeddings)
    assert new_preds_compat == orig_preds
    np.testing.assert_allclose(new_scores_compat, orig_scores, rtol=1e-5)
    print("Verification successful: Backward compatibility works.")

    # Test empty input
    empty_scores, empty_preds = estimator.predict(np.array([]).reshape(0, D))
    assert empty_scores == []
    assert empty_preds == []
    print("Verification successful: Empty input handled.")

if __name__ == "__main__":
    verify()
