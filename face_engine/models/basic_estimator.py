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
        # Pre-calculate squared norms for faster distance computation in predict
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Ensure embeddings is a numpy array
        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        # Vectorized distance calculation using expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is significantly faster than loop-based np.linalg.norm
        q_norms_sq = np.sum(embeddings**2, axis=1, keepdims=True)
        f_norms_sq = getattr(self, "norms_sq", None)
        if f_norms_sq is None:
            # Fallback for models fitted with older versions
            f_norms_sq = np.sum(self.embeddings**2, axis=1)

        dists_sq = (
            q_norms_sq + f_norms_sq - 2 * np.dot(embeddings, self.embeddings.T)
        )
        # Handle numerical precision issues (e.g. slightly negative values)
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
