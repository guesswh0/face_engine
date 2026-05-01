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
        self.norms_sq = np.sum(np.square(self.embeddings), axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Vectorized distance computation using ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is significantly faster than iterative approach for large datasets.
        a = embeddings
        b = self.embeddings

        a_norms_sq = np.sum(np.square(a), axis=1)
        b_norms_sq = getattr(self, "norms_sq", None)
        if b_norms_sq is None:
            # Fallback for models fitted with older versions
            b_norms_sq = np.sum(np.square(b), axis=1)

        # dists_sq shape: (n_predict, n_fitted)
        dists_sq = (
            a_norms_sq[:, np.newaxis] + b_norms_sq[np.newaxis, :] - 2 * np.dot(a, b.T)
        )

        indices = np.argmin(dists_sq, axis=1)
        min_dists_sq = dists_sq[np.arange(len(a)), indices]

        # Ensure we don't have negative values due to floating point noise
        min_dists_sq = np.maximum(min_dists_sq, 0)

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
