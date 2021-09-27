"""
FaceEngine core module.
"""

import os
import pickle

import numpy as np

from . import logger
from .exceptions import FaceNotFoundError
from .models import _models
from .tools import imread


def load_engine(filename):
    """Loads and restores engine object from the file.

    This function is convenient wrapper of pickle.load() function, and is used
    to deserialize and restore the engine object from the persisted state.

    Estimator model's state is loaded separately and is loaded only
    if there is something saved before by :meth:`~FaceEngine.save` method.
    Estimator model serialization (.save) and deserialization (.load) process
    steps are the responsibility of it's inheriting class.

    :param filename: serialized by :meth:`~FaceEngine.save` method file name
    :type filename: str

    :return: restored engine object
    :rtype: :class:`.FaceEngine`
    """

    with open(filename, 'rb') as file:
        engine = pickle.load(file)

    # foolproof
    if not isinstance(engine, FaceEngine):
        raise TypeError("file %s could not be deserialized as "
                        "FaceEngine instance" % filename)

    # load estimator model's state only if there is something to restore
    if engine.n_samples > 0:
        # pass filename's directory name
        engine._estimator.load(os.path.dirname(filename))
    return engine


class FaceEngine:
    """Face recognition engine base class.

    This object provides all steps and tools which are required to work with
    face recognition task.

    Keyword arguments:
        * detector (str) -- face detector model to use
        * embedder (str) -- face embedder model to use
        * estimator (str) -- face estimator model to use
    """

    def __init__(self, **kwargs):
        """Create new FaceEngine instance"""

        # computation core trio
        self.detector = kwargs.get('detector')
        self.embedder = kwargs.get('embedder')
        self.estimator = kwargs.get('estimator')
        # keep last fitted number of classes and samples
        self.n_classes = 0
        self.n_samples = 0

    def __getstate__(self):
        # copy the engine object's state from self.__dict__
        # using copy to avoid modifying the original state
        state = self.__dict__.copy()

        # remove the model objects (unpicklable entries)
        del state['_detector']
        del state['_embedder']
        del state['_estimator']

        # persist the model objects names
        state['detector'] = self.detector
        state['embedder'] = self.embedder
        state['estimator'] = self.estimator

        # returns engine instance's lightweight state dictionary
        # contents of the which will be used by pickle in .save() method
        return state

    def __setstate__(self, state):
        # initialize engine models by their setter methods
        self.detector = state.pop('detector')
        self.embedder = state.pop('embedder')
        self.estimator = state.pop('estimator')

        # update rest attributes
        self.__dict__.update(state)

    @property
    def detector(self):
        """
        :return: detector model name
        :rtype: str
        """

        return self._detector.name

    @detector.setter
    def detector(self, name):
        """Face detector model to use:
            - 'hog': dlib "Histogram Oriented Gradients" model (default).
            - 'mmod': dlib "Max-Margin Object Detection" model.

        :param name: detector model name
        :type name: str
        """

        if not name:
            name = 'hog'
        if name not in _models:
            if name != 'hog':
                logger.warning("Detector model '%s' not found!", name)
            name = 'abstract_detector'
        Detector = _models.get(name)
        self._detector = Detector()

    @property
    def embedder(self):
        """
        :return: embedder model name
        :rtype: str
        """

        return self._embedder.name

    @embedder.setter
    def embedder(self, name):
        """Face embedder model to use:
            - 'resnet': dlib ResNet model (default)

        :param name: embedder model name
        :type name: str
        """

        if not name:
            name = 'resnet'
        if name not in _models:
            if name != 'resnet':
                logger.warning("Embedder model '%s' not found!", name)
            name = 'abstract_embedder'
        Embedder = _models.get(name)
        self._embedder = Embedder()

    @property
    def estimator(self):
        """
        :return: estimator model name
        :rtype: str
        """

        return self._estimator.name

    @estimator.setter
    def estimator(self, name):
        """Estimator model to use:
            - 'basic': linear comparing estimator (default)

        :param name: estimator model name
        :type name: str
        """

        if not name:
            name = 'basic'
        if name not in _models:
            if name != 'basic':
                logger.warning("Estimator model '%s' not found!", name)
            name = 'abstract_estimator'
        Estimator = _models.get(name)
        self._estimator = Estimator()

    def save(self, filename):
        """Save engine object state to the file.

        Persisting the object state as lightweight engine instance which
        contains only model name strings instead of model objects itself.
        Upon loading model objects will have to be re-initialized.
        """

        # save estimator model's state only if there is something to persist
        if self.n_samples > 0:
            self._estimator.save(os.path.dirname(filename))

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def _fit(self, embeddings, class_names, **kwargs):
        """Fit (train) estimator model with given embeddings for
        given class names.

        :param embeddings: face embedding vectors
            with shape (n_samples, embedding_dim)
        :type embeddings: numpy.ndarray

        :param class_names: sequence of class names
        :type class_names: list

        :keyword kwargs: model and data dependent

        :return: self

        :raises: TrainError
        """

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # may raise TrainError
        self._estimator.fit(embeddings, class_names, **kwargs)
        self.n_samples = len(embeddings)
        self.n_classes = len(set(class_names))

        return self

    def fit(self, images, class_names, **kwargs):
        """Fit (train) estimator model with given images for
        given class names.

        Estimator's :meth:`~face_engine.models.Estimator.fit` wrapping method.

        .. note::
            * the number of images and class_names has to be equal
            * the image will be skipped if the face is not found inside

        :param images: image file names or uri strings
        :type images: list[str]

        :param class_names: sequence of class names
        :type class_names: list

        :keyword kwargs: estimator model and data dependent

        :return: self

        :raises: TrainError
        """

        assert len(images) == len(class_names), (
            "the number of images and class_names has to be equal")

        targets = list()
        embeddings = list()

        for image, target in zip(images, class_names):
            img = imread(image)
            try:
                # find largest face in the image
                bb, extra = self.find_faces(img, limit=1)
                embedding = self.compute_embeddings(img, bb, **extra)[0]
                targets.append(target)
                embeddings.append(embedding)
            except FaceNotFoundError:
                # if face not found in the image, skip it
                continue
        embeddings = np.array(embeddings)
        return self._fit(embeddings, targets, **kwargs)

    def predict(self, embeddings):
        """Make predictions for given embeddings.

        Estimator's :meth:`~face_engine.models.Estimator.predict`
        wrapping method.

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.ndarray

        :returns: prediction scores and class names
        :rtype: (list, list)

        :raises: TrainError
        """

        return self._estimator.predict(embeddings)

    def make_prediction(self, image, **kwargs):
        """Lazy prediction method to make prediction by given image.

        Convenient wrapper method to go over all steps of face recognition
        problem by one call.

        In particular:
            1. :meth:`.find_faces` - detector
            2. :meth:`.compute_embeddings` - embedder
            3. :meth:`.predict` - estimator

        Keyword arguments are all parameters of :meth:`.find_faces` method.
        Returns image all face bounding boxes with predicted class names.
        May raise same exceptions of all calling methods.

        :param image: RGB image content or image file uri.
        :type image: Union[str, bytes, file, os.PathLike, numpy.ndarray]

        :returns: bounding boxes and class_names
        :rtype: tuple(list, list)

        :raises: FaceNotFoundError
        :raises: TrainError
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        bounding_boxes, extra = self.find_faces(image, **kwargs)
        embeddings = self.compute_embeddings(image, bounding_boxes, **extra)
        class_names = self.predict(embeddings)[1]
        return bounding_boxes, class_names

    def find_faces(self, image, limit=None, normalize=False):
        """Find multiple faces in the image.

        Detector's :meth:`~face_engine.models.Detector.detect_all`
        wrapping method.

        :param image: RGB image content or image file uri.
        :type image: Union[str, bytes, file, os.PathLike, numpy.ndarray]

        :param limit: limit the number of detected faces on the image
            by bounding box size.
        :type limit: int

        :param normalize: normalize output bounding boxes
        :type normalize: bool

        :returns: face bounding box with shape (n_faces, 4),
            detector model dependent extra information.
        :rtype: (numpy.ndarray, dict)

        :raises: FaceNotFoundError
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        # original image height and width
        height, width = image.shape[0:2]

        bbs, extra = self._detector.detect(image)

        n_det = len(bbs)
        if isinstance(limit, int) and limit < n_det:
            if self.detector in ['hog', 'mmod']:
                indices = range(limit)
            else:
                indices = np.argsort(
                    [(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in bbs]
                )[::-1][:limit]
            # limit extra fields if any exist
            for key, value in extra.items():
                extra[key] = extra[key][limit]
            bbs = bbs[indices]

        if normalize:
            bbs = bbs / ([width, height] * 2)
        return bbs, extra

    def compute_embeddings(self, image, bounding_boxes, **kwargs):
        """Compute image embeddings for given bounding boxes.

        Embedder's :meth:`~face_engine.models.Embedder.compute_embeddings`
        wrapping method.

        :param image: RGB image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: face bounding boxes
        :type bounding_boxes: numpy.ndarray

        :keyword kwargs: model dependent

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        embeddings = self._embedder.compute_embeddings(
            image, bounding_boxes, **kwargs)
        return embeddings
