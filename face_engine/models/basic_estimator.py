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
        self._sq_norm_fitted = None

    def fit(self, embeddings, class_names, **kwargs):
        # Convert to numpy array to ensure vectorized operations work
        self.embeddings = np.asarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms of fitted embeddings for optimization
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        X = np.asarray(embeddings)
        if X.size == 0:
            return [], []

        # Vectorized distance calculation using (a-b)^2 = a^2 + b^2 - 2ab
        # This is significantly faster than the iterative approach for large datasets.
        x_sq_norm = np.sum(X**2, axis=1, keepdims=True)
        # self._sq_norm_fitted is pre-calculated in fit()
        dists_sq = x_sq_norm + self._sq_norm_fitted - 2 * (X @ self.embeddings.T)

        # Ensure no negative values due to floating point precision errors
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(X)), indices]

        scores = np.exp(-0.5 * min_dists_sq)
        predicted_classes = [self.class_names[i] for i in indices]

        return scores.tolist(), predicted_classes

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))

        # Backward compatibility: recalculate squared norms if missing from loaded state
        if self.embeddings is not None and self._sq_norm_fitted is None:
            self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)
