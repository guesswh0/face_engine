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
        # Pre-calculate squared norms of fitted embeddings for faster distance calculation in predict
        self.norms_sq = np.sum(self.embeddings**2, axis=1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        embeddings = np.asarray(embeddings)

        # Vectorized distance calculation using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        # query_norms_sq shape: (n_query,)
        query_norms_sq = np.sum(embeddings**2, axis=1)

        # dot_product shape: (n_query, n_fitted)
        dot_product = np.dot(embeddings, self.embeddings.T)

        # distances_sq shape: (n_query, n_fitted)
        # Using broadcasting: (n_query, 1) + (n_fitted,) - 2 * (n_query, n_fitted)
        # We use getattr for self.norms_sq to maintain backward compatibility with older saved models
        fitted_norms_sq = getattr(self, 'norms_sq', None)
        if fitted_norms_sq is None:
            fitted_norms_sq = np.sum(self.embeddings**2, axis=1)

        distances_sq = query_norms_sq[:, np.newaxis] + fitted_norms_sq - 2 * dot_product

        # Ensure distances are non-negative (can happen due to floating point errors)
        distances_sq = np.maximum(distances_sq, 0)

        # Find index of minimum distance for each query
        indices = np.argmin(distances_sq, axis=1)

        # Calculate scores and get class names
        # min_distances_sq shape: (n_query,)
        min_distances_sq = distances_sq[np.arange(len(embeddings)), indices]
        scores = np.exp(-0.5 * min_distances_sq).tolist()
        predicted_classes = [self.class_names[i] for i in indices]

        return scores, predicted_classes

    def save(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "wb") as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = "%s.estimator.%s" % (self.name, "p")
        with open(os.path.join(dirname, name), "rb") as file:
            self.__dict__.update(pickle.load(file))
