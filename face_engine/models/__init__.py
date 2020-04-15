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

__all__ = ['models', 'Model', 'Detector', 'Embedder', 'Predictor']

from face_engine.tools import import_submodules

models = {}
"""storage for registered model classes"""


class Model:
    """FaceEngine model base class. Used to register all inheriting and
    imported subclasses (subclass registration #PEP487).

        - implementing model classes must have `name` class descriptor

    """

    name = None
    """short model name of implementing class"""

    def __set_name__(self, owner, name):
        print(self.name, owner, name)

    def __init_subclass__(cls, name=None, **kwargs):
        if name:
            cls.name = name
            models[name] = cls
        elif cls.__name__ in __all__:
            cls.name = 'abstract_' + cls.__name__.lower()
            models[cls.name] = cls
        super().__init_subclass__(**kwargs)


class Detector(Model):
    """Human face detector object.

        -   bounding box format is (left, top, right, bottom)
    """

    def detect_all(self, image):
        """Detect all face bounding boxes in the image,
         with corresponding confidence scores.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence scores and bounding boxes.
             shape of bounding box is (n_faces, 4).
        :rtype: tuple(numpy.ndarray, numpy.ndarray)

        :raises FaceError: if there is no faces in the image
        """

        raise NotImplementedError()

    def detect_one(self, image):
        """Detect the image largest face bounding box.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence score and bounding box.
        :rtype: numpy.ndarray

        :raises FaceError: if there is no faces in the image
        """

        raise NotImplementedError()


class Embedder(Model):
    """This object calculates embedding vectors from the face containing image.

        -   implementing model classes must have `dim` class descriptor
    """

    def __init_subclass__(cls, name=None, dim=None, **kwargs):
        cls.embedding_dim = dim
        super().__init_subclass__(name, **kwargs)

    def compute_embedding(self, image, bounding_box):
        """Compute image embedding for given bounding box

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_box: face bounding box
        :type bounding_box: list | numpy.ndarray

        :returns: embedding vector
        :rtype: numpy.ndarray
        """

        raise NotImplementedError()

    def compute_embeddings(self, image, bounding_boxes):
        """Compute image embeddings for given bounding boxes

        :param image: RGB image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: face bounding boxes with shape (n_faces, 4).
        :type bounding_boxes: numpy.ndarray

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        raise NotImplementedError()


class Predictor(Model):
    """This object is used to make predictions, to which class input face
    embeddings belongs to, with some prediction score"""

    def init_model(self, embedding_dim=None):
        """Initialize and build predictor model (if required).
        @override if predictor model requires separate method to init itself.

        :param embedding_dim: optional
        :type embedding_dim: int
        """

    def fit(self, embeddings, class_names):
        """Fit predictor with given embeddings for given class names

        :param embeddings: face embedding vectors
            with shape (n_samples, embedding_dim)
        :type embeddings: numpy.ndarray

        :param class_names: sequence of class names
        :type class_names: list | numpy.ndarray

        :returns self: object

        :raises TrainError: if model fit (train) fails
        """

        raise NotImplementedError()

    def predict(self, embeddings):
        """Predict class name by given embeddings.

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.array

        :returns: prediction scores and class names
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        :raises TrainError: if model not fitted
        """

        raise NotImplementedError()

    def save(self, path):
        """Persist predictor model state to given path"""

        raise NotImplementedError()

    def load(self, path):
        """Load predictor model state from given path"""

        raise NotImplementedError()


import_submodules(__file__)
