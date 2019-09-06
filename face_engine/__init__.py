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

import os

import numpy as np
from skimage import io
from skimage.transform import rescale

from .exceptions import FaceError, TrainError
from .models import Detector, Embedder, Predictor

__version__ = '1.0.0'

BASE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(BASE)
RESOURCES = os.path.join(BASE, 'resources')


class FaceEngine:
    """Face recognition engine object.

    Project main purpose is to simplify work with `face recognition problem`
    computation core trio - detector, embedder, and predictor. FaceEngine
    combines all of them in one interface model to simplify usage and
    furthermore extends some features.

    FaceEngine is working out of the box, with pre-defined default models. But
    it is possible (and sometimes required) to use your own models for detector,
    embedder or predictor. FaceEngine is designed the way, when you can easily
    plugin your own model. All you need to do is to implement model interfaces
    Detector, Embedder or Predictor (see `models` package), `register`
    model (import) and `create` instance of it with `use_plugin` method.
    """

    def __init__(self, limit=1000):
        """ All models is defined by their default values, to change model
        use corresponding setter method.

        Examples:
            to change model to dlib mmod detector use:
                >>>> engine = FaceEngine()
                >>>> engine.detector = 'mmod'

            to import and use your own plugin model use:
                >>>> engine = FaceEngine()
                >>>> engine.use_plugin(
                >>>>    Detector, 'mmod', 'face_engine/models/mmod_detector.py')

        :param limit: is required to restrict the number of faces fed
            to predictor
        :type limit: int

        """
        self.limit = limit
        # computation core trio
        self.detector = None
        self.embedder = None
        self.predictor = None
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
        Detector.register(name)
        cls = Detector.models.get(name)
        self._detector = Detector.create(subclass=cls)

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
        Embedder.register(name)
        cls = Embedder.models.get(name)
        self._embedder = Embedder.create(subclass=cls)

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

            -   'linear': linear comparing by calculating L2-norms (default)

        :param name: predictor model name
        :type name: str
        """

        if not name:
            name = 'linear'
        Predictor.register(name)
        cls = Predictor.models.get(name)
        self._predictor = Predictor.create(cls)

    def use_plugin(self, model_type, name, filepath, **kwargs):
        """Used to register > create instance > set attribute for self-defined
        plugin models.

        Plugin model is required to follow rules of model_type, which is
        defined in `models` package, to make it work. See default examples in
        models directory.

        :param model_type: model type

        :param name: model name
        :type name: str

        :param filepath: absolute or relative filepath to plugin model file
        :type filepath: str

        :param kwargs: additional kwargs required to pass to the plugin model
            __init__ method.
        """

        model_type.register(filepath, plugin=True)
        cls = model_type.models.get(name)
        model = model_type.create(cls, **kwargs)
        setattr(self, cls.suffix, model)

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
        :type bounding_boxes: list | numpy.ndarray

        :returns: self

        :raises TrainError: if model fit (train) fails
        or numbers of samples exceeds buffer size
        """

        targets = list()
        embeddings = list()

        if bounding_boxes:
            for image, bb in zip(images, bounding_boxes):
                img = io.imread(image)
                embedding = self._embedder.compute_embedding(img, bb)
                embeddings.append(embedding)
            targets = class_names
        else:
            for image, target in zip(images, class_names):
                img = io.imread(image)
                try:
                    _, bb = self._detector.detect_one(img)
                    embedding = self._embedder.compute_embedding(img, bb)
                    targets.append(target)
                    embeddings.append(embedding)
                except FaceError:
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

    def predict(self, image_or_embeddings):
        """`Predictor`s wrapping method to predict class name by given image or
        embedding vectors. If fed with image, default values for `find_faces`
        and `compute_embeddings` will be used.

        [*] may raise FaceError only if had fed with image.

        :param image_or_embeddings: RGB image or array of embedding
        vectors with shape (n_faces, embedding_dim)
        :type image_or_embeddings: numpy.array

        :returns: predicted similarity scores and class names
        :rtype: tuple[numpy.ndarray, numpy.ndarray]

        :raises TrainError: if model not fitted
        :raises FaceError: if there is no faces in the image.
        """

        # check if arg is image
        if len(image_or_embeddings.shape) > 2:
            _, bounding_boxes = self.find_faces(image_or_embeddings)
            embeddings = self.compute_embeddings(image_or_embeddings,
                                                 bounding_boxes)
            scores, class_names = self._predictor.predict(embeddings)
        else:
            scores, class_names = self._predictor.predict(image_or_embeddings)
        return scores, class_names

    def find_face(self, image, scale=None, normalize=False):
        """Find one face in the image. 'Detector's wrapping method.
        Used to find the image largest face bounding box.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding box

        :returns confidence score and bounding box
        :rtype tuple[numpy.ndarray, numpy.ndarray]

        :raises FaceError: if there is no face in the image
        """

        img_size = np.asarray(image.shape)[0:2]
        if scale:
            rescaled_img = rescale(
                image, (scale, scale), preserve_range=True, multichannel=True
            ).astype(np.uint8)
            # detect and scale back
            confidence, bounding_box = self._detector.detect_one(rescaled_img)
            bounding_box = (bounding_box / scale).astype(np.uint16)
            # just in case bind to image size
            bounding_box[0] = np.maximum(bounding_box[0], 0)
            bounding_box[1] = np.maximum(bounding_box[1], 0)
            bounding_box[2] = np.minimum(bounding_box[2], img_size[1])
            bounding_box[3] = np.minimum(bounding_box[3], img_size[0])
        else:
            confidence, bounding_box = self._detector.detect_one(image)

        if normalize:
            bounding_box = bounding_box.astype(np.float32)
            bounding_box[0] = bounding_box[0] / img_size[1]
            bounding_box[1] = bounding_box[1] / img_size[0]
            bounding_box[2] = bounding_box[2] / img_size[1]
            bounding_box[3] = bounding_box[3] / img_size[0]
        return confidence, bounding_box

    def find_faces(self, image, borders=None, scale=None, normalize=False):
        """ Find multiple faces in the image. 'Detector's wrapping method.
            Used to find faces bounding boxes of in the image, with several
            pre and post-processing abilities.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param borders: area of interest borders
            i.e two points of rectangle (top, left, bottom, right)
        :type borders: tuple | list

        :param scale: scale image by a certain factor, value > 0
        :type scale: float

        :param normalize: normalize output bounding boxes
        :type normalize: bool

        :returns: confidence scores and bounding boxes
        :rtype tuple[numpy.ndarray, numpy.ndarray]

        :raises FaceError: if there is no faces in the image
        """

        img_size = np.asarray(image.shape)[0:2]
        # crop image by borders
        if borders:
            image = image[borders[1]:borders[3], borders[0]:borders[2], :]

        if scale:
            rescaled_img = rescale(
                image, (scale, scale), preserve_range=True, multichannel=True
            ).astype(np.uint8)

            # detect and scale back
            confidences, bounding_boxes = \
                self._detector.detect_all(rescaled_img)
            bounding_boxes = (bounding_boxes / scale).astype(np.uint16)
            # just in case bind to image size
            bounding_boxes[:, 0] = np.maximum(bounding_boxes[:, 0], 0)
            bounding_boxes[:, 1] = np.maximum(bounding_boxes[:, 1], 0)
            bounding_boxes[:, 2] = np.minimum(bounding_boxes[:, 2], img_size[1])
            bounding_boxes[:, 3] = np.minimum(bounding_boxes[:, 3], img_size[0])
        else:
            confidences, bounding_boxes = self._detector.detect_all(image)

        # adopt bounding box to original image borders
        if borders:
            bounding_boxes += np.array(borders[:2] * 2, dtype=np.uint16)

        if normalize:
            bounding_boxes = bounding_boxes.astype(np.float32)
            bounding_boxes[:, 0] = bounding_boxes[:, 0] / img_size[1]
            bounding_boxes[:, 1] = bounding_boxes[:, 1] / img_size[0]
            bounding_boxes[:, 2] = bounding_boxes[:, 2] / img_size[1]
            bounding_boxes[:, 3] = bounding_boxes[:, 3] / img_size[0]
        return confidences, bounding_boxes

    def compute_embeddings(self, image, bounding_boxes):
        """`Embedder`s wrapping method. Used to compute
        image embeddings for given bounding boxes.

        :param image: RGB Image with shape (rows, cols, 3)
        :type image: numpy.ndarray

        :param bounding_boxes: bounding boxes
        :type bounding_boxes: numpy.ndarray

        :returns: array of embedding vectors with shape (n_faces, embedding_dim)
        :rtype: numpy.ndarray
        """

        return self._embedder.compute_embeddings(image, bounding_boxes)

    def load(self, filename):
        """Load model state - helper method"""

        import pickle
        from pathlib import PurePosixPath

        with open(filename, 'rb') as file:
            model_state = pickle.load(file)
        self.__dict__.update(model_state)
        self.detector = model_state['detector']
        self.embedder = model_state['embedder']
        self.predictor = model_state['predictor']

        pps = PurePosixPath(filename)
        self._predictor.load(str(pps.parent))

    def save(self, filename):
        """Save model state - helper method"""

        import pickle
        from pathlib import PurePosixPath

        _copy = self.__dict__.copy()
        # cleanup and reassign models by their names
        del _copy['_detector']
        _copy['detector'] = self.detector
        del _copy['_embedder']
        _copy['embedder'] = self.embedder
        del _copy['_predictor']
        _copy['predictor'] = self.predictor

        # save
        pps = PurePosixPath(filename)
        self._predictor.save(str(pps.parent))
        with open(filename, 'wb') as file:
            pickle.dump(_copy, file)
