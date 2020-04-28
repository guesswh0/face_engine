"""
FaceEngine models API.
"""

__all__ = ['Detector', 'Embedder', 'Estimator',
           '_models', 'compare', 'BasicEstimator']

import os
import pickle

import numpy as np

from face_engine.exceptions import TrainError
from face_engine.tools import import_package

_models = {}
"""storage for all registered model classes"""


class Model:
    """FaceEngine model base class. Used to register all inheriting and
    imported subclasses (subclass registration #PEP487).

    Note:
        - implementing model classes must have `name` class descriptor

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

    Note:
        - bounding box format is (left, upper, right, lower)
    """

    def detect_all(self, image):
        """Detect all face bounding boxes in the image,
         with corresponding confidence scores.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence scores and bounding boxes.
        :rtype: tuple(list, list[tuple])

        :raise FaceNotFoundError
        """

        raise NotImplementedError()

    def detect_one(self, image):
        """Detect the image largest face bounding box.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :returns: confidence score and bounding box.
        :rtype: tuple(float, tuple)

        :raise FaceNotFoundError
        """

        raise NotImplementedError()


class Embedder(Model):
    """This object calculates embedding vectors from the face containing image.

    Note:
        - implementing model classes should have `dim` class descriptor
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

        Note that number samples of passed embbedings and class_names
        has to be equal.

        :param embeddings: face embedding vectors
            with shape (n_samples, embedding_dim)
        :type embeddings: numpy.ndarray

        :param class_names: sequence of class names
        :type class_names: list

        :keyword kwargs: model and data dependent

        :returns: self

        :raise TrainError
        """

        raise NotImplementedError()

    def predict(self, embeddings):
        """Make predictions for given embeddings.

        To call predict(), model previously has to be fitted (trained).

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.array

        :returns: prediction scores and class names
        :rtype: tuple(list, list)

        :raise TrainError
        """

        raise NotImplementedError()

    def save(self, dirname):
        """Persist estimators's model state to given directory.

        File naming format convention:
            name = '%s.estimator.%s' % (self.name, ext)
        """

        raise NotImplementedError()

    def load(self, dirname):
        """Restore estimator's model state from given directory.

        File naming format convention:
            name = '%s.estimator.%s' % (self.name, ext)
        """

        raise NotImplementedError()


def compare(source, target):
    """Compare vectors OvR. Returns the most similar target vector
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


class BasicEstimator(Estimator, name='basic'):
    """Basic estimator model, make predictions by linear comparing each source
    embedding vector with each fitted embedding vectors.

    Model is using python pickle module to persist estimator state. Default
    file name is 'basic.estimator.p'.

    (*) This model is used as FaceEngine default estimator.
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
            index, score = compare(embedding, self.embeddings)
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


import_package(__file__)
