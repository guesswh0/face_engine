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
        # Pre-calculate squared norms for vectorized distance calculation
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Vectorized implementation using squared distance expansion:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab^T
        # This significantly speeds up batch processing by avoiding Python loops.

        # A: (n_query, dim), B: (n_fitted, dim)
        A = np.asarray(embeddings)
        B = self.embeddings

        # A_sq: (n_query, 1)
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        # B_sq: (1, n_fitted)
        B_sq = getattr(self, "norms_sq", None)
        if B_sq is None:
            B_sq = np.sum(B**2, axis=1)

        # AB_T: (n_query, n_fitted)
        AB_T = np.dot(A, B.T)

        # dists_sq: (n_query, n_fitted)
        dists_sq = A_sq + B_sq - 2 * AB_T

        # Clip to 0 for numerical stability (handle small negative values due to floating point error)
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(A)), indices]

        scores = np.exp(-0.5 * min_dists_sq).tolist()
        predicted_classes = [self.class_names[i] for i in indices]

        return scores, predicted_classes

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))
