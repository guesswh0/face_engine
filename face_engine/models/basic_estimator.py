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
        self.fitted_norms_sq = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = embeddings
        self.class_names = class_names
        # Pre-calculate squared norms for faster distance computation in predict
        self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        embeddings = np.asarray(embeddings)
        if embeddings.size == 0:
            return [], []

        # Vectorized Euclidean distance calculation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is much faster than looping over each embedding.
        if self.fitted_norms_sq is None:
            self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

        query_norms_sq = np.sum(embeddings**2, axis=1)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # dists_sq shape: (n_queries, n_fitted)
        dists_sq = (
            self.fitted_norms_sq + query_norms_sq[:, np.newaxis] - 2 * dot_product
        )
        # Handle potential negative values due to floating point precision
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(embeddings)), indices]

        scores = np.exp(-0.5 * min_dists_sq)
        predicted_names = [self.class_names[i] for i in indices]

        return scores.tolist(), predicted_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))

        # Reconstruct fitted_norms_sq if it's missing from the saved state
        if self.embeddings is not None and (
            not hasattr(self, "fitted_norms_sq") or self.fitted_norms_sq is None
        ):
            self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1)
