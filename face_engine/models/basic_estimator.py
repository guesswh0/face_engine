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
        self.norms_sq = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = np.asarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms for faster distance calculation
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        # Using expansion formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is much faster than looping and using np.linalg.norm
        q_norms_sq = np.sum(embeddings**2, axis=1, keepdims=True)

        # Handle backward compatibility for models fitted with older versions
        fitted_norms_sq = getattr(self, "norms_sq", None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

        # Calculate squared Euclidean distances
        dists_sq = (
            q_norms_sq
            + fitted_norms_sq
            - 2 * np.dot(embeddings, self.embeddings.T)
        )

        # Numerical stability: ensure distances are non-negative
        dists_sq = np.maximum(dists_sq, 0)

        # Find best matches
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
