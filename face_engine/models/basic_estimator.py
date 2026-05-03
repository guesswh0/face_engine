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

        if len(embeddings) == 0:
            return [], []

        # Vectorized implementation using squared Euclidean distance expansion:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab

        # Calculate squared norms of input embeddings: (M, 1)
        input_norms_sq = np.sum(embeddings**2, axis=1, keepdims=True)

        # Get or calculate squared norms of fitted embeddings: (1, N)
        # Using getattr for backward compatibility with older saved models
        fitted_norms_sq = getattr(self, "norms_sq", None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

        # Compute dot product: (M, N)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # Compute squared distances: (M, N)
        dists_sq = input_norms_sq + fitted_norms_sq - 2 * dot_product

        # Ensure no negative values due to floating point inaccuracies
        dists_sq = np.maximum(dists_sq, 0)

        # Find nearest neighbor for each input embedding
        indices = np.argmin(dists_sq, axis=1)

        # Extract minimum squared distances and compute scores
        min_dists_sq = dists_sq[np.arange(len(embeddings)), indices]
        scores = np.exp(-0.5 * min_dists_sq)

        # Map indices to class names
        predicted_class_names = [self.class_names[i] for i in indices]

        return scores.tolist(), predicted_class_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))
