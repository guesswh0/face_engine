import os
import pickle

import numpy as np

from face_engine.exceptions import TrainError
from face_engine.models import Estimator


class BasicEstimator(Estimator, name="basic"):
    """Basic estimator model makes predictions by linear comparing each source
    embedding vector with each fitted embedding vectors.

    Model is using python pickle module to persist estimator state. Default
    file name is ``'basic.estimator.p'``.
    """

    def __init__(self):
        self.embeddings = None
        self.class_names = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = np.asanyarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms of fitted embeddings for faster prediction
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Ensure inputs are numpy arrays
        A = np.asanyarray(embeddings)
        B = self.embeddings

        if len(A) == 0:
            return [], []

        # Vectorized distance calculation using the expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        # A: (N, D), B: (M, D)
        sq_norm_A = np.sum(A**2, axis=1, keepdims=True)  # (N, 1)

        # Use pre-calculated norms if available, otherwise calculate on the fly
        sq_norm_B = getattr(self, "_sq_norm_fitted", None)
        if sq_norm_B is None:
            sq_norm_B = np.sum(B**2, axis=1)  # (M,)

        # Matrix multiplication A @ B.T gives dot products for all pairs (N, M)
        # dists_sq: (N, M)
        dists_sq = sq_norm_A + sq_norm_B - 2 * (A @ B.T)

        # Numerical stability: distances squared should be >= 0
        dists_sq = np.maximum(dists_sq, 0)

        # Find nearest neighbor indices and minimum squared distances
        indices = np.argmin(dists_sq, axis=1)
        min_d2 = dists_sq[np.arange(len(A)), indices]

        # Calculate scores and retrieve class names
        scores = np.exp(-0.5 * min_d2).tolist()
        class_names = [self.class_names[i] for i in indices]

        return scores, class_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))

        # Backward compatibility: if _sq_norm_fitted is missing, calculate it
        if self.embeddings is not None and getattr(self, "_sq_norm_fitted", None) is None:
            self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)
