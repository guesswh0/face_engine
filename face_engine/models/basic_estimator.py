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
        # Pre-calculate squared norms for faster prediction
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Vectorized calculation of squared Euclidean distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>

        # input_norms_sq: (M, 1)
        input_norms_sq = np.sum(embeddings**2, axis=1, keepdims=True)

        # fitted_norms_sq: (1, N)
        # Handle backward compatibility for models loaded from older versions
        fitted_norms_sq = getattr(self, "norms_sq", None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(self.embeddings**2, axis=1)
        fitted_norms_sq = fitted_norms_sq.reshape(1, -1)

        # dot_product: (M, N)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # dists_sq: (M, N)
        # Using np.maximum to prevent small negative values from floating point errors
        dists_sq = np.maximum(input_norms_sq + fitted_norms_sq - 2 * dot_product, 0)

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
