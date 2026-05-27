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
        # Pre-calculated norms for vectorized distance calculation
        self.fitted_norms_sq = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = embeddings
        self.class_names = class_names
        # Pre-calculating norms to speed up prediction
        self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        embeddings = np.asarray(embeddings)
        if embeddings.size == 0:
            return [], []

        # Vectorized Euclidean distance using the expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is significantly faster than calculating norm in a loop.
        b2 = np.sum(embeddings**2, axis=1)
        ab = np.dot(self.embeddings, embeddings.T)

        # dists_sq shape: (n_fitted, n_queries)
        dists_sq = self.fitted_norms_sq[:, np.newaxis] + b2 - 2 * ab

        # Avoid negative values due to floating point precision issues
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=0)
        min_dists_sq = dists_sq[indices, np.arange(len(embeddings))]

        scores = np.exp(-0.5 * min_dists_sq)
        class_names = [self.class_names[i] for i in indices]

        return list(scores), class_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))

        # Ensure backward compatibility if fitted_norms_sq was not saved
        if self.embeddings is not None and self.fitted_norms_sq is None:
            self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1)
