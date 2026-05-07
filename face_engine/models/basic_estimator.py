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
        # Pre-calculate squared norms for faster distance calculation in predict
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        embeddings = np.asarray(embeddings)
        if len(embeddings) == 0:
            return [], []

        # Vectorized distance calculation using the expansion:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*<a, b>

        # Calculate squared norms of input embeddings: (N, 1)
        input_norms_sq = np.sum(embeddings**2, axis=1)[:, np.newaxis]

        # Matrix multiplication for dot products: (N, M)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # Get fitted norms: (1, M)
        # Using getattr for backward compatibility with older saved models
        fitted_norms_sq = getattr(self, "norms_sq", None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

        # Broadcasted sum to get squared distances: (N, M)
        dists_sq = input_norms_sq + fitted_norms_sq - 2 * dot_product

        # Handle potential numerical precision issues (e.g., small negative values)
        dists_sq = np.maximum(dists_sq, 0)

        # Find minimum distance index for each input embedding
        indices = np.argmin(dists_sq, axis=1)

        # Extract the minimum squared distances
        min_dists_sq = dists_sq[np.arange(len(embeddings)), indices]

        # Calculate scores and retrieve class names
        scores = np.exp(-0.5 * min_dists_sq).tolist()
        predicted_class_names = [self.class_names[i] for i in indices]

        return scores, predicted_class_names

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))
