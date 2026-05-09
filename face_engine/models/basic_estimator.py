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
        self.embeddings = np.asarray(embeddings)
        self.class_names = class_names
        # Pre-calculate squared norms of fitted embeddings for faster distance computation
        self.norms_sq = np.sum(np.square(self.embeddings), axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Ensure embeddings is a numpy array
        embeddings = np.asarray(embeddings)

        # Vectorized distance calculation using the expansion formula:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        # This is significantly faster than looping and using np.linalg.norm.

        # Get pre-calculated squared norms of fitted embeddings
        # Use getattr for backward compatibility with models fitted in older versions
        norms_sq_fitted = getattr(self, "norms_sq", None)
        if norms_sq_fitted is None:
            norms_sq_fitted = np.sum(np.square(self.embeddings), axis=1)

        norms_sq_input = np.sum(np.square(embeddings), axis=1)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # dists_sq shape: (n_input, n_fitted)
        dists_sq = norms_sq_input[:, np.newaxis] + norms_sq_fitted - 2 * dot_product

        # Clip to 0 to avoid tiny negative values due to floating point precision
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
