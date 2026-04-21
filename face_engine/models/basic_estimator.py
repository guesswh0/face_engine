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
        self.embeddings = embeddings
        self.class_names = class_names
        # Optimization: Pre-calculate squared norms of fitted embeddings
        # to speed up vectorized distance calculation in predict method.
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Ensure backward compatibility if loaded from an older model state
        if not hasattr(self, "_sq_norm_fitted"):
            self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

        # Optimization: Use squared distance expansion formula to vectorize calculation:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This avoids expensive Python loops and significantly improves performance.
        sq_norm_input = np.sum(embeddings**2, axis=1)
        # Using np.dot for efficient matrix multiplication
        dists_sq = (
            self._sq_norm_fitted
            + sq_norm_input[:, np.newaxis]
            - 2 * np.dot(embeddings, self.embeddings.T)
        )

        # Handle potential negative values due to floating point precision
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(indices)), indices]

        scores = np.exp(-0.5 * min_dists_sq).tolist()
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
