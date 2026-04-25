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
        self.embeddings = np.asanyarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms for faster distance calculation in predict
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        embeddings = np.asanyarray(embeddings)

        # Vectorized distance calculation using the expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        a_sq = np.sum(embeddings**2, axis=1)[:, np.newaxis]

        # Use pre-calculated norms if available, otherwise calculate on the fly
        b_sq = getattr(self, "_sq_norm_fitted", None)
        if b_sq is None:
            b_sq = np.sum(self.embeddings**2, axis=1)

        ab = np.dot(embeddings, self.embeddings.T)

        # dists_sq shape: (n_input, n_fitted)
        dists_sq = a_sq + b_sq - 2 * ab
        # Handle potential negative values due to floating point precision errors
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
