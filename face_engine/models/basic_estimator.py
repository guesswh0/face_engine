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
        self.embeddings = np.asanyarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms of fitted embeddings for faster distance calculation
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        embeddings = np.asanyarray(embeddings)

        # Vectorized distance calculation using the expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is significantly faster than looping through each input embedding.
        sq_norm_input = np.sum(embeddings**2, axis=1)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # Resulting dists_sq has shape (n_input, n_fitted)
        dists_sq = sq_norm_input[:, np.newaxis] + self._sq_norm_fitted - 2 * dot_product

        # Handle floating point precision errors that might lead to tiny negative values
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        # Extract the minimum squared distances for each input embedding
        min_dists_sq = dists_sq[np.arange(len(embeddings)), indices]

        scores = np.exp(-0.5 * min_dists_sq).tolist()
        class_names = [self.class_names[idx] for idx in indices]

        return scores, class_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))

        # Re-calculate pre-calculated norms if they are missing from the loaded state
        if self.embeddings is not None and self._sq_norm_fitted is None:
            self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)
