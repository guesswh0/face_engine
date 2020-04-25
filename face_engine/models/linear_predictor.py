# Copyright 2020 Daniyar Kussainov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

import numpy as np

from face_engine.exceptions import TrainError
from face_engine.models import Predictor


def compare(source, target):
    """ Compare vectors OvR. Returns the most similar target vector
    index and score value.

    Compares source vector with each target vectors by calculating
    euclidean distances (L2-norms).

    Similarity score is estimated with RBF kernel function.

    References:
        [1] https://en.wikipedia.org/wiki/Euclidean_distance
        [2] https://en.wikipedia.org/wiki/Radial_basis_function_kernel

    :param source: source vector of shape (vector_dim,)
    :type source: numpy.ndarray

    :param target: target vectors of shape (n_samples, vector_dim)
    :type target: numpy.ndarray

    :returns: index and similarity score
    :rtype: tuple(int, float)
    """

    distances = np.linalg.norm(target - source, axis=1)
    index = np.argmin(distances)
    score = np.exp(-0.5 * distances[index] ** 2)
    return index, score


class LinearPredictor(Predictor, name='linear'):
    """ Linear predictor model. Makes predictions by linear comparing each
    source embedding vector with each fitted embedding vectors.
    """

    def __init__(self):
        self.embeddings = None
        self.class_names = None

    def fit(self, embeddings, class_names):
        self.embeddings = embeddings
        self.class_names = class_names

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError('Model is not fitted yet!')

        scores = []
        class_names = []
        for embedding in embeddings:
            index, score = compare(embedding, self.embeddings)
            scores.append(score)
            class_names.append(self.class_names[index])
        return scores, class_names

        """

        distances = np.linalg.norm(target - source, axis=1)
        index = np.argmin(distances)
        score = np.exp(-0.5 * distances[index] ** 2)
        return index, score

    def save(self, path):
        name = self.name + '.pkl'
        with open(os.path.join(path, name), 'wb') as file:
            pickle.dump(self.__dict__, file)

    def load(self, path):
        name = self.name + '.pkl'
        with open(os.path.join(path, name), 'rb') as file:
            self.__dict__.update(pickle.load(file))
