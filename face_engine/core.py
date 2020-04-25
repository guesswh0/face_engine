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

import numpy as np
from PIL import Image

from . import logger
from .exceptions import FaceError, TrainError
from .models import _models
from .tools import imread


class FaceEngine:
    """Face recognition engine object.

    Project main purpose is to simplify work with `face recognition problem`
    computation core trio - detector, embedder, and predictor. FaceEngine
    combines all of them in one interface model to simplify usage and
    furthermore extends some features.

    FaceEngine is working out of the box, with pre-defined default models. But
    if you want to, you can work with your own pre-trained models for detector,
    embedder or predictor. All you need to do is to implement model interfaces
    Detector, Embedder or Predictor (see `models` package), and "register" it,
    by importing your module or adding it to `PYTHONPATH` environment variable
    or using appropriate convenient function `from face_engine.tools`.

    From here you can use your model just by passing model name with
    corresponding keyword argument of `__init__` method or setup it later by
    calling corresponding setter method of FaceEngine object with model name
    argument.

    Examples:
        To change model to dlib 'mmod' detector use:
            >>> from face_engine import FaceEngine
            >>> engine = FaceEngine()
            >>> engine.detector = 'mmod'

        To switch to your own model use corresponding setter method:
            >>> from my_custom_models import my_custom_detector
            >>> engine.detector = 'custom_detector'

        To init with your own pre-trained detector use:
            >>> from my_custom_models import my_custom_detector
            >>> engine = FaceEngine(detector='custom_detector')


    Keyword arguments:
        :param detector: face detector model to use
        :type detector: str

        :param embedder: face embedder model to use
        :type embedder: str

        :param predictor: face predictor model to use
        :type predictor: str

        :param limit: limit number of faces fed to predictor
        :type limit: int
    """

    def __init__(self, **kwargs):
        """Create new FaceEngine instance"""

        self.limit = kwargs.get('limit', 1000)
        # computation core trio
        self.detector = kwargs.get('detector')
        self.embedder = kwargs.get('embedder')
        self.predictor = kwargs.get('predictor')
        # keep last fitted number of identities and samples
        self.n_identities = 0
        self.n_samples = 0

    @property
    def detector(self):
        """
        :returns: detector name
        :rtype: str
        """

        return self._detector.name

    @detector.setter
    def detector(self, name):
        """Face detector model to use:

            -   'hog': dlib "Histogram Oriented Gradients" model (default).
            -   'mmod': dlib "Max-Margin Object Detection" model.

        :param name: detector model name
        :type name: str
        """

        if not name:
            name = 'hog'
        if name not in _models:
            logger.warning("Detector model '%s' not found!", name)
            name = 'abstract_detector'
        model = _models.get(name)
        self._detector = model()

    @property
    def embedder(self):
        """
        :returns: embedder model name
        :rtype: str
        """

        return self._embedder.name

    @embedder.setter
    def embedder(self, name):
        """Face embedder model to use:

            -   'resnet': dlib ResNet model (default)

        :param name: embedder model name
        :type name: str
        """

        if not name:
            name = 'resnet'
        if name not in _models:
            logger.warning("Embedder model '%s' not found!", name)
            name = 'abstract_embedder'
        model = _models.get(name)
        self._embedder = model()

    @property
    def predictor(self):
        """
        :returns: predictor name
        :rtype: str
        """

        return self._predictor.name

    @predictor.setter
    def predictor(self, name):
        """Face predictor model to use:

            -   'linear': linear comparing, by calculating `L2-norms` with
            RBF kernel function (default)

        :param name: predictor model name
        :type name: str
        """

        if not name:
            name = 'linear'
        if name not in _models:
            logger.warning("Predictor model '%s' not found!", name)
            name = 'abstract_predictor'
        model = _models.get(name)
        self._predictor = model()

    def fit(self, images, class_names, bounding_boxes=None):
        """Fit face predictor model with given images for given class names.

            -   if 'bounding_boxes' presents skips bounding_box detections

        ------------------------------------------------------------------------

        [*] To not to use large memory buffers (image arrays) using filenames
            or url strings.

        [*] Expensive operation. Depends on the numbers of samples, and
            bounding box presence.

        :param images: filenames or URLs of images
        :type images: list[str]

        :param class_names: sequence of class names
        :type class_names: list | numpy.ndarray

        :param bounding_boxes: sequence of bounding boxes
        :type bounding_boxes: list[tuple]

        :returns: self

        :raises TrainError: if model fit (train) fails
        or numbers of samples exceeds buffer size
        """

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
                except FaceError:
                    # if face not found in image skip it
                    continue

        n_samples = len(targets)
        n_identities = len(set(targets))
        embeddings = np.array(embeddings)
        targets = np.array(targets)

        if n_samples > self.limit:
            raise TrainError('Enlarge buffer size')

        # also may raise TrainError
        self._predictor.fit(embeddings, targets)

        self.n_samples = n_samples
        self.n_identities = n_identities
        return self

    def predict(self, embeddings):
        """Predict class name by given embedding vectors.

        Predictor's wrapping method

        :param embeddings: array of embedding vectors
            with shape (n_faces, embedding_dim)
        :type embeddings: numpy.ndarray

        :returns: prediction scores and class names
        :rtype: tuple(list, list)

        :raises TrainError: if model not fitted
        """

        return self._predictor.predict(embeddings)

    def make_prediction(self, image, **kwargs):
        """ Lazy prediction method to make prediction by given image.

        Convenient wrapper method to go over all steps of face recognition
        problem by one call.

        In particular:
            .find_faces() - detector
            .compute_embeddings() - embedder
            .predict() - predictor

        Keyword arguments is all parameters of .find_faces() method.
        Returns image all face bounding boxes with predicted class names.
        May raise same exceptions of all calling methods.

        :param image: actual image content or image file uri.
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :returns: bounding boxes, and class names
        :rtype: tuple(list, list)

        :raises FaceError: if there is no faces in the image
        :raises TrainError: if model not fitted
        """

        if not hasattr(image, 'shape'):
            image = imread(image)

        _, bounding_boxes = self.find_faces(image, **kwargs)
        embeddings = self.compute_embeddings(image, bounding_boxes)
        _, class_names = self.predict(embeddings)
        return bounding_boxes, class_names

    def find_face(self, image, scale=None, normalize=False):
        """Find one face in the image. 'Detector's wrapping method.
        Used to find the image largest face bounding box.

        :param image: RGB image, content or file uri.
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding box

        :returns confidence score and bounding box
        :rtype tuple(float, tuple)

        :raises FaceError: if there is no face in the image
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
        """Find multiple faces in the image. Detector's wrapping method.
        Used to find faces bounding boxes of in the image, with several
        pre and post-processing abilities.

        :param image: RGB image, content or file uri.
        :type image: numpy.ndarray | {str, bytes, file, os.PathLike}

        :param roi: region of interest rectangle,
            format (left, upper, right, lower)
        :type roi: tuple | list

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding boxes
        :type normalize: bool

        :returns: confidence scores and bounding boxes
        :rtype tuple(list, list[tuple])

        :raises FaceError: if there is no faces in the image
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
        """Embedders wrapping method.
        Compute image embedding for given bounding box.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_box: face bounding box
        :type bounding_box: tuple

        :returns: embedding vector
        :rtype: numpy.ndarray
        """

        return self._embedder.compute_embedding(image, bounding_box)

    def compute_embeddings(self, image, bounding_boxes):
        """Embedders wrapping method.
        Compute image embeddings for given bounding boxes.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: face bounding boxes
        :type bounding_boxes: list[tuple]

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        return self._embedder.compute_embeddings(image, bounding_boxes)

    def load(self, filename):
        """Load model state - helper method"""

        import pickle

        with open(filename, 'rb') as file:
            model_state = pickle.load(file)
        self.__dict__.update(model_state)
        self.detector = model_state['detector']
        self.embedder = model_state['embedder']
        self.predictor = model_state['predictor']

        # load predictor state
        self._predictor.load(os.path.dirname(filename))

    def save(self, filename):
        """Save model state - helper method"""

        import pickle

        _copy = self.__dict__.copy()
        # cleanup and reassign models by their names
        del _copy['_detector']
        _copy['detector'] = self.detector
        del _copy['_embedder']
        _copy['embedder'] = self.embedder
        del _copy['_predictor']
        _copy['predictor'] = self.predictor

        # save predictor state
        self._predictor.save(os.path.dirname(filename))
        with open(filename, 'wb') as file:
            pickle.dump(_copy, file)
