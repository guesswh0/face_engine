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
        # Pre-calculate squared norms of fitted embeddings for faster distance
        # calculation in the predict method.
        self.norms_sq = np.sum(self.embeddings**2, axis=1, keepdims=True).T

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        embeddings = np.asarray(embeddings)

        # Vectorized distance calculation using expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # Resulting dists_sq has shape (n_predict, n_fitted)
        emb_sq = np.sum(embeddings**2, axis=1, keepdims=True)

        # Handle models loaded from old pickle files that don't have norms_sq
        norms_sq = getattr(self, "norms_sq", None)
        if norms_sq is None:
            norms_sq = np.sum(self.embeddings**2, axis=1, keepdims=True).T

        dists_sq = emb_sq + norms_sq - 2 * np.dot(embeddings, self.embeddings.T)

        # Numerical stability: distances can't be negative
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
