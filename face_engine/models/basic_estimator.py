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
        # Pre-calculate squared norms of fitted embeddings for faster prediction
        self._sq_norm_fitted = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Vectorized implementation using the squared distance expansion:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>
        # This significantly speeds up batch predictions by reducing redundant calculations.
        dot_product = np.dot(embeddings, self.embeddings.T)
        sq_norm_query = np.sum(embeddings**2, axis=1, keepdims=True)

        # sq_distances is (n_faces, n_samples)
        sq_distances = np.maximum(sq_norm_query + self._sq_norm_fitted - 2 * dot_product, 0)
        distances = np.sqrt(sq_distances)

        # Find the index of the closest fitted embedding for each input face
        indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(embeddings)), indices]

        # Calculate scores and retrieve class names
        scores = np.exp(-0.5 * min_distances**2).tolist()
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
