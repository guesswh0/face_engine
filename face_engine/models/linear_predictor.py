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


class LinearPredictor(Predictor, name='linear'):
    """Linear predictor calculates `euclidean distance` (L2-norm) and
    applies RBF kernel function `exp(-1/2*||x-x'||^2)` with `sigma=1`.

        -   output prediction (similarity) score is in range (0,1).
    """

    def __init__(self):
        self.embeddings = None
        self.class_names = None

    def fit(self, embeddings, class_names):
        self.embeddings = embeddings
        self.class_names = class_names

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError('Model not fitted yet!')

        n_faces = embeddings.shape[0]
        scores = np.empty(n_faces, dtype=np.float32)
        indices = np.empty(n_faces, dtype=np.uint32)
        for i, embedding in enumerate(embeddings):
            indices[i], scores[i] = self.compare(embedding, self.embeddings)
        return scores, self.class_names[indices]

    @staticmethod
    def compare(source, target):
        """Convenient method to compare embeddings OvR.

        :param source: source embedding, of shape (emb_dim, )
        :type source: numpy.array

        :param target: target embeddings of shape (n_samples, embedding_dim)
        :type target: numpy.array

        :returns: similarity score, and class name
        :rtype: tuple[float, int]

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
