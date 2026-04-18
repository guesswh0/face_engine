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

    def fit(self, embeddings, class_names, **_kwargs):
        # Ensure embeddings is a NumPy array for optimized distance calculations
        self.embeddings = (
            embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings)
        )
        self.class_names = class_names

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Ensure input embeddings is a NumPy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Optimized vectorized distance calculation using squared distance expansion:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # M: number of input embeddings, N: number of fitted embeddings
        # embeddings: (M, D), self.embeddings: (N, D)

        # (M,)
        a_sq = np.sum(np.square(embeddings), axis=1)
        # (N,)
        b_sq = np.sum(np.square(self.embeddings), axis=1)
        # (M, N)
        ab = np.dot(embeddings, self.embeddings.T)

        # (M, N) using broadcasting: (M, 1) + (N,) - 2 * (M, N)
        sq_distances = a_sq[:, np.newaxis] + b_sq - 2 * ab
        # Ensure no negative values due to numerical instability
        sq_distances = np.maximum(sq_distances, 0)

        # Find indices of minimum distances for each input embedding
        indices = np.argmin(sq_distances, axis=1)
        min_sq_distances = sq_distances[np.arange(len(embeddings)), indices]

        # Calculate scores and retrieve class names
        scores = np.exp(-0.5 * min_sq_distances).tolist()
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
