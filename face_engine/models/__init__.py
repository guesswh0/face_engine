# Copyright 2019 Daniyar Kussainov
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

"""
This package implements `face recognition problem` computations core trio -
detector, embedder and predictor as plugin based abstract classes. All three
objects implement the same behaviour to register (import) and initialize
subclasses (#PEP487).
"""

import importlib


class PluginMixin:
    """ This mixin class defines the rules for plugin model registering and
    instance creating for all inheriting model classes.

        -   all implementing plugin model classes must have `name` class
        attribute (descriptor).
    """

    name = None
    """short model name of implementing class"""

    suffix = None
    """convenient attribute, used for registering default defined models"""

    models = {}
    """plugin models dictionary, where to store registered classes"""

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            cls.name = name
            cls.models[name] = cls

    @classmethod
    def register(cls, name, plugin=False):
        if plugin:
            from importlib import util
            from pathlib import PurePosixPath
            pps = PurePosixPath(name)
            spec = util.spec_from_file_location(pps.stem, name)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            importlib.import_module('.' + name + cls.suffix, __name__)

    @staticmethod
    def create(subclass, **kwargs):
        if subclass:
            return subclass(**kwargs)


class Detector(PluginMixin):
    """Human face detector object.

        -   bounding box format is (top, left, right, bottom)
    """

    suffix = '_detector'

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


class Embedder(PluginMixin):
    """This object calculates embedding vectors from the face containing image.

        -   inheriting plugin classes additionally must have `dim` class
     attribute (descriptor), which describes embedder output vector dimensions.
    """

    suffix = '_embedder'

    def __init_subclass__(cls, name=None, dim=None, **kwargs):
        super().__init_subclass__(name, **kwargs)
        cls.embedding_dim = dim

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


class Predictor(PluginMixin):
    """This object is used to make predictions, which class input face
    embeddings belongs to, with some prediction score"""

    suffix = '_predictor'

    def init_model(self, embedding_dim=None):
        """Initialize and build predictor model (if required).
        @override if predictor model requires separate method to init itself.

        :param embedding_dim: optional
        :type embedding_dim: int
        """
        pass

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
