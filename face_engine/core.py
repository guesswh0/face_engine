"""
FaceEngine core module.
"""

import os
import pickle

import numpy as np
from PIL import Image

from . import logger
from .exceptions import FaceNotFoundError, TrainError
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
        * limit (int) -- limit number of faces fed to estimator
    """

    def __init__(self, **kwargs):
        """Create new FaceEngine instance"""

        self.limit = kwargs.get('limit', 1000)
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
        :type embeddings: numpy.ndarray | list

        :param class_names: sequence of class names
        :type class_names: list

        :keyword kwargs: model and data dependent

        :return: self

        :raises: TrainError
        """

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if len(embeddings) > self.limit:
            raise TrainError('Enlarge buffer size')

        # also may raise TrainError
        self._estimator.fit(embeddings, class_names, **kwargs)
        self.n_samples = len(embeddings)
        self.n_classes = len(set(class_names))

        return self

    def fit(self, images, class_names, bounding_boxes=None, **kwargs):
        """Fit (train) estimator model with given images for
        given class names.

        Estimator's :meth:`~face_engine.models.Estimator.fit` wrapping method.

        .. note::
            * the number of images and class_names has to be equal
            * the image will be skipped if the face is not found inside
            * the presence of 'bounding_boxes' accelerates process

        [*] Uses array of file names or uri strings instead of large
        memory buffers (image arrays).

        :param images: image file names or uri strings
        :type images: list[str]

        :param class_names: sequence of class names
        :type class_names: list

        :param bounding_boxes: sequence of bounding boxes
        :type bounding_boxes: list[tuple]

        :keyword kwargs: model and data dependent

        :return: self

        :raises: TrainError
        """

        assert len(images) == len(class_names), (
            "the number of images and class_names has to be equal")

        targets = list()
        embeddings = list()

        if bounding_boxes:
            for image, bb in zip(images, bounding_boxes):
                img = imread(image)
                embedding = self._embedder.compute_embedding(img, bb)
                embeddings.append(embedding)
            targets = class_names
        else:
            for image, target in zip(images, class_names):
                img = imread(image)
                try:
                    _, bb = self._detector.detect_one(img)
                    embedding = self._embedder.compute_embedding(img, bb)
                    targets.append(target)
                    embeddings.append(embedding)
                except FaceNotFoundError:
                    # if face not found in the image, skip it
                    continue

        return self._fit(embeddings, targets, **kwargs)

    def predict(self, embeddings):
        """Make predictions for given embeddings.

        Estimator's :meth:`~face_engine.models.Estimator.predict`
        wrapping method.

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.ndarray

        :returns: prediction scores and class names
        :rtype: tuple(list, list)

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
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :returns: class names and bounding boxes
        :rtype: tuple(list, list)

        :raises: FaceNotFoundError
        :raises: TrainError
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        bounding_boxes = self.find_faces(image, **kwargs)[1]
        embeddings = self.compute_embeddings(image, bounding_boxes)
        class_names = self.predict(embeddings)[1]
        return class_names, bounding_boxes

    def find_face(self, image, scale=None, normalize=False):
        """Find one face in the image.

        .. note::
           If the image contains multiple faces, detects image
           largest face bounding box.

        Detector's :meth:`~face_engine.models.Detector.detect_one`
        wrapping method.

        :param image: RGB image content or image file uri.
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding box
        :type normalize: bool

        :returns: confidence score and bounding box
        :rtype: tuple(float, tuple)

        :raises: FaceNotFoundError
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        # original image height and width
        height, width = image.shape[0:2]

        if scale:
            size = (int(width * scale), int(height * scale))
            image = np.uint8(Image.fromarray(image).resize(size))
            confidence, bounding_box = self._detector.detect_one(image)
            # scale bounding_box to original image size
            bounding_box = (max(bounding_box[0] // scale, 0),
                            max(bounding_box[1] // scale, 0),
                            min(bounding_box[2] // scale, width),
                            min(bounding_box[3] // scale, height))
        else:
            confidence, bounding_box = self._detector.detect_one(image)

        if normalize:
            bounding_box = (bounding_box[0] / width,
                            bounding_box[1] / height,
                            bounding_box[2] / width,
                            bounding_box[3] / height)
        return confidence, bounding_box

    def find_faces(self, image, roi=None, scale=None, normalize=False):
        """Find multiple faces in the image.

        Used to find all faces bounding boxes in the image.

        Detector's :meth:`~face_engine.models.Detector.detect_all`
        wrapping method.

        :param image: RGB image content or image file uri.
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :param roi: region of interest rectangle,
            format (left, upper, right, lower)
        :type roi: tuple | list

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding boxes
        :type normalize: bool

        :returns: confidence scores and bounding boxes
        :rtype: tuple(list, list[tuple])

        :raises: FaceNotFoundError
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        # original image height and width
        height, width = image.shape[0:2]

        # crop image by region of interest
        if roi:
            image = image[roi[1]:roi[3], roi[0]:roi[2], :]

        if scale:
            h, w = image.shape[0:2]
            size = (int(w * scale), int(h * scale))
            image = np.uint8(Image.fromarray(image).resize(size))
            confidences, bounding_boxes = self._detector.detect_all(image)
            # scale back bounding_boxes to image size
            bounding_boxes = [(
                max(int(bounding_box[0] / scale), 0),
                max(int(bounding_box[1] / scale), 0),
                min(int(bounding_box[2] / scale), w),
                min(int(bounding_box[3] / scale), h))
                for bounding_box in bounding_boxes]
        else:
            confidences, bounding_boxes = self._detector.detect_all(image)

        # adopt bounding box to original image size
        if roi:
            from operator import add

            roi = roi[:2] * 2
            bounding_boxes = [
                tuple(map(add, bounding_box, roi))
                for bounding_box in bounding_boxes]

        if normalize:
            bounding_boxes = [(
                bounding_box[0] / width,
                bounding_box[1] / height,
                bounding_box[2] / width,
                bounding_box[3] / height)
                for bounding_box in bounding_boxes]
        return confidences, bounding_boxes

    def compute_embedding(self, image, bounding_box):
        """Compute image embedding for given bounding box.

        Embedder's :meth:`~face_engine.models.Embedder.compute_embedding`
        wrapping method.

        :param image: RGB image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_box: face bounding box
        :type bounding_box: tuple

        :return: embedding vector
        :rtype: numpy.ndarray
        """

        return self._embedder.compute_embedding(image, bounding_box)

    def compute_embeddings(self, image, bounding_boxes):
        """ Compute image embeddings for given bounding boxes.

        Embedder's :meth:`~face_engine.models.Embedder.compute_embeddings`
        wrapping method.

        :param image: RGB image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: face bounding boxes
        :type bounding_boxes: list[tuple]

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        return self._embedder.compute_embeddings(image, bounding_boxes)

    def compare_faces(self, source, target):
        """Compare a face in the source image with each face in the
        target image, to find out the most similar one.

        .. note::
           If the source image contains multiple faces, detects image
           largest face bounding box.

        Similarity score is estimated with RBF kernel.

        References:
            1. https://en.wikipedia.org/wiki/Euclidean_distance
            2. https://en.wikipedia.org/wiki/Radial_basis_function_kernel

        :param source: RGB image content or image file uri.
        :type source: numpy.ndarray | {str, bytes, file, os.PathLike}

        :param target: RGB image content or image file uri.
        :type target: numpy.ndarray | {str, bytes, file, os.PathLike}

        :returns: similarity score and bounding box
        :rtype: tuple(float, tuple)
        """

        if not hasattr(source, 'shape'):
            source = imread(source)
        source_bb = self._detector.detect_one(source)[1]
        source_vector = self._embedder.compute_embedding(source, source_bb)

        if not hasattr(target, 'shape'):
            target = imread(target)
        target_bbs = self._detector.detect_all(target)[1]
        target_vector = self._embedder.compute_embeddings(target, target_bbs)

        distances = np.linalg.norm(target_vector - source_vector, axis=1)
        index = np.argmin(distances)
        score = np.exp(-0.5 * distances[index] ** 2)
        return score, target_bbs[index]
