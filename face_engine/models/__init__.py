"""
FaceEngine models API.
"""

__all__ = ['Detector', 'Embedder', 'Estimator', '_models']

import numpy as np

from face_engine.tools import import_package

_models = {}
"""storage for all registered model classes"""


class Model:
    """FaceEngine model base class. Used to register all inheriting and
    imported subclasses (subclass registration PEP 487).

    .. note::
        * implementing model classes must have ``name`` class descriptor

    """

    name = None
    """short model name of implementing class"""

    def __set_name__(self, owner, name):
        print(self.name, owner, name)

    def __init_subclass__(cls, name=None, **kwargs):
        if name:
            cls.name = name
            _models[name] = cls
        elif cls.__name__ in __all__:
            cls.name = 'abstract_' + cls.__name__.lower()
            _models[cls.name] = cls
        super().__init_subclass__(**kwargs)


class Detector(Model):
    """Human face detector model base class.

    .. note::
        * bounding box format is (left, upper, right, lower)
    """

    def detect_all(self, image):
        """Detect all face bounding boxes in the image, with corresponding
        confidence scores.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence scores and bounding boxes.
        :rtype: tuple(list, list[tuple])

        :raises: FaceNotFoundError
        """

        raise NotImplementedError()

    def detect_one(self, image):
        """Detect the image largest face bounding box.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence score and bounding box.
        :rtype: tuple(float, tuple)

        :raises: FaceNotFoundError
        """

        raise NotImplementedError()


class Embedder(Model):
    """This object calculates embedding vectors from the face containing image.

    .. note::
        * implementing model classes should have ``dim`` class descriptor
    """

    def __init_subclass__(cls, name=None, dim=None, **kwargs):
        cls.embedding_dim = dim
        super().__init_subclass__(name, **kwargs)

    def compute_embedding(self, image, bounding_box):
        """Compute image embedding for given bounding box

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_box: face bounding box
        :type bounding_box: tuple

        :returns: embedding vector
        :rtype: numpy.ndarray
        """

        raise NotImplementedError()

    def compute_embeddings(self, image, bounding_boxes):
        """Compute image embeddings for given bounding boxes

        :param image: RGB image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: face bounding boxes
        :type bounding_boxes: list[tuple]

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        raise NotImplementedError()


class Estimator(Model):
    """Estimator model base class. Used to make predictions for face
    embedding vectors.
    """

    def fit(self, embeddings, class_names, **kwargs):
        """Fit (train) estimator model with given embeddings for
        given class names.

        Note that the passed number of samples for ``embbedings`` and
        ``class_names`` has to be equal.

        :param embeddings: face embedding vectors
            with shape (n_samples, embedding_dim)
        :type embeddings: numpy.ndarray

        :param class_names: sequence of class names
        :type class_names: list

        :keyword kwargs: model and data dependent

        :returns: self

        :raises: TrainError
        """

        raise NotImplementedError()

    def predict(self, embeddings):
        """Make predictions for given embeddings.

        .. note::
           Model previously has to be fitted.

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.array

        :returns: prediction scores and class names
        :rtype: tuple(list, list)

        :raises: TrainError
        """

        raise NotImplementedError()

    def save(self, dirname):
        """Persist estimators's model state to given directory.

        File naming format convention:
          ``name = '%s.estimator.%s' % (self.name, ext)``
        """

        raise NotImplementedError()

    def load(self, dirname):
        """Restore estimator's model state from given directory.

        File naming format convention:
            ``name = '%s.estimator.%s' % (self.name, ext)``
        """

        raise NotImplementedError()


import_package(__file__)
