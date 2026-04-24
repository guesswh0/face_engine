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
        self.embeddings = np.asarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms of fitted embeddings to speed up prediction
        # Using squared norm avoids sqrt and speeds up distance calculations
        self._sq_norm_fitted = np.sum(np.square(self.embeddings), axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Vectorized prediction using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        # This is much faster than iterating through input embeddings.
        # It also maintains O(M x N) memory efficiency by avoiding large broadcasting.
        # where M is n_predict and N is n_fitted.

        # Ensure input is a numpy array
        embeddings = np.asarray(embeddings)

        # Calculate ||a||^2 for input embeddings
        sq_norm_input = np.sum(np.square(embeddings), axis=1)

        # Calculate 2*a*b using dot product
        # dot_product shape: (n_predict, n_fitted)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # dists_sq shape: (n_predict, n_fitted)
        # Using broadcasting: (n_predict, 1) + (1, n_fitted) - (n_predict, n_fitted)
        dists_sq = sq_norm_input[:, np.newaxis] + self._sq_norm_fitted - 2 * dot_product

        # Clean up floating point errors (distances shouldn't be negative)
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(embeddings)), indices]

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

        # Backward compatibility: re-calculate squared norms if they are missing
        if self.embeddings is not None and getattr(self, "_sq_norm_fitted", None) is None:
            self._sq_norm_fitted = np.sum(np.square(self.embeddings), axis=1)
