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
        self.fitted_norms_sq = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = embeddings
        self.class_names = class_names
        # Precompute squared norms for faster distance calculation in predict()
        # Shape: (n_samples, 1)
        self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1).reshape(-1, 1)

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        # Ensure input is a numpy array for vectorized operations
        embeddings = np.asarray(embeddings)
        if embeddings.size == 0:
            return [], []

        # Vectorized Euclidean distance calculation using the expansion formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # This significantly reduces complexity from O(N*M*D) to matrix operations
        # which are highly optimized in NumPy/BLAS.
        # N = number of query embeddings, M = number of fitted embeddings, D = dimension

        # self.fitted_norms_sq is (M, 1)
        # query_norms_sq is (1, N)
        query_norms_sq = np.sum(embeddings**2, axis=1).reshape(1, -1)

        # dists_sq shape will be (M, N)
        # Formula: ||a||^2 + ||b||^2 - 2*a.dot(b.T)
        dists_sq = self.fitted_norms_sq + query_norms_sq - 2 * self.embeddings.dot(embeddings.T)

        # Clip very small negative values that might occur due to floating point precision
        dists_sq = np.maximum(dists_sq, 0)

        indices = np.argmin(dists_sq, axis=0)  # (N,)

        # Extract minimum squared distances for score calculation
        min_dists_sq = np.take_along_axis(dists_sq, indices[np.newaxis, :], axis=0).flatten()

        # Calculate scores using the formula: exp(-0.5 * dist^2)
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

        # Recompute norms for backward compatibility with models saved before this optimization
        if self.embeddings is not None and self.fitted_norms_sq is None:
            self.fitted_norms_sq = np.sum(self.embeddings**2, axis=1).reshape(-1, 1)
