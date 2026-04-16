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

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        if len(embeddings) == 0:
            return [], []

        # Vectorized distance calculation using the formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        a2 = np.sum(embeddings**2, axis=1, keepdims=True)
        b2 = np.sum(self.embeddings**2, axis=1)
        ab = np.dot(embeddings, self.embeddings.T)
        dist2 = np.maximum(a2 + b2 - 2 * ab, 0)

        indices = np.argmin(dist2, axis=1)
        # score = exp(-0.5 * dist^2)
        scores = np.exp(-0.5 * dist2[np.arange(len(embeddings)), indices]).tolist()
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
