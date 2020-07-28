import os
import pickle

import numpy as np

from face_engine.exceptions import TrainError
from face_engine.models import Estimator


class BasicEstimator(Estimator, name='basic'):
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
            raise TrainError('Model is not fitted yet!')

        scores = []
        class_names = []
        for embedding in embeddings:
            distances = np.linalg.norm(self.embeddings - embedding, axis=1)
            index = np.argmin(distances)
            score = np.exp(-0.5 * distances[index] ** 2)
            scores.append(score)
            class_names.append(self.class_names[index])
        return scores, class_names

    def save(self, dirname):
        name = '%s.estimator.%s' % (self.name, 'p')
        with open(os.path.join(dirname, name), 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, dirname):
        name = '%s.estimator.%s' % (self.name, 'p')
        with open(os.path.join(dirname, name), 'rb') as file:
            self.__dict__.update(pickle.load(file))
