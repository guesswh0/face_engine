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

        embeddings = np.atleast_2d(embeddings)

        # Vectorized distance calculation using the expansion:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # This is significantly faster than np.linalg.norm in a loop.

        # Use getattr for backward compatibility with older persisted models
        fitted_norms_sq = getattr(self, "norms_sq", None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(np.square(self.embeddings), axis=1)

        input_norms_sq = np.sum(np.square(embeddings), axis=1)

        # -2ab term
        dot_product = np.dot(embeddings, self.embeddings.T)

        # ||a||^2 + ||b||^2 - 2ab
        # Using broadcasting:
        # (n_input, 1) + (n_fitted,) - (n_input, n_fitted)
        dists_sq = input_norms_sq[:, np.newaxis] + fitted_norms_sq - 2 * dot_product

        # Ensure non-negative due to floating point inaccuracies
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
