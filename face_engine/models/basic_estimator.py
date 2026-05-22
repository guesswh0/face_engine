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
        # Pre-calculate squared norms of fitted embeddings for vectorized distance computation
        self.fitted_norms_sq = np.sum(np.square(self.embeddings), axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Guard clause for empty input to avoid dimension errors
        if len(embeddings) == 0:
            return [], []

        # Vectorized distance calculation using the expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is much faster than looping and using np.linalg.norm for each query
        query_embeddings = np.asarray(embeddings)
        query_norms_sq = np.sum(np.square(query_embeddings), axis=1)
        dot_product = np.dot(query_embeddings, self.embeddings.T)

        # dists_sq shape: (n_queries, n_fitted)
        dists_sq = (
            query_norms_sq[:, np.newaxis] + self.fitted_norms_sq - 2 * dot_product
        )

        # Handle potential small negative values due to floating-point precision
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

        # Reconstruct fitted_norms_sq for backward compatibility with models
        # saved before the vectorization optimization.
        if self.embeddings is not None and self.fitted_norms_sq is None:
            self.fitted_norms_sq = np.sum(np.square(self.embeddings), axis=1)
