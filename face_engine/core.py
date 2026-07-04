"""
FaceEngine core module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os

import numpy as np

from . import __version__, logger
from .exceptions import FaceNotFoundError
from .models import _models
from .tools import imread

# default model fallback chains: insightface -> dlib -> abstract
_DETECTOR_DEFAULTS = ("scrfd", "hog")
_EMBEDDER_DEFAULTS = ("arcface", "resnet")
_ESTIMATOR_DEFAULTS = ("basic",)


def _resolve(name: Optional[str], defaults: Tuple[str, ...], kind: str) -> str:
    """Resolve model name to a registered one, with installed-backend
    fallbacks for empty names and abstract fallback for unknown names."""

    if not name:
        for candidate in defaults:
            if candidate in _models:
                return candidate
        return "abstract_" + kind
    if name not in _models:
        logger.warning("%s model '%s' not found!", kind.capitalize(), name)
        return "abstract_" + kind
    return name


_LEGACY_PICKLE_ERROR = (
    "%s appears to be a legacy pickle produced by face-engine < 3.0. "
    "Pickle persistence was removed in 3.0.0 for security. "
    "Re-create the engine, re-fit it, then save() again."
)


def load_engine(filename: str) -> "FaceEngine":
    """Loads and restores engine object from the file.

    Restores the engine object from the JSON state persisted by
    :meth:`~FaceEngine.save` method.

    Estimator model's state is loaded separately and is loaded only
    if there is something saved before by :meth:`~FaceEngine.save` method.
    Estimator model serialization (.save) and deserialization (.load) process
    steps are the responsibility of it's inheriting class.

    :param filename: serialized by :meth:`~FaceEngine.save` method file name
    :type filename: str

    :return: restored engine object
    :rtype: :class:`.FaceEngine`

    :raises RuntimeError: on legacy (pre-3.0) pickle files
    """

    with open(filename, "rb") as file:
        content = file.read()

    # pre-3.0 engines were pickled (protocol >= 2 starts with b'\x80')
    if content[:1] == b"\x80":
        raise RuntimeError(_LEGACY_PICKLE_ERROR % filename)
    try:
        data = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError(
            "file %s could not be deserialized as FaceEngine state" % filename
        )

    # foolproof
    if not isinstance(data, dict) or data.get("format") != "face-engine":
        raise TypeError(
            "file %s could not be deserialized as FaceEngine state" % filename
        )

    engine = FaceEngine(
        detector=data.get("detector"),
        embedder=data.get("embedder"),
        estimator=data.get("estimator"),
    )
    engine.n_classes = data.get("n_classes", 0)
    engine.n_samples = data.get("n_samples", 0)
    for key, value in data.get("extra", {}).items():
        setattr(engine, key, value)

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

    def __init__(self, **kwargs: Any) -> None:
        """Create new FaceEngine instance"""

        # computation core trio
        self.detector = kwargs.get("detector")
        self.embedder = kwargs.get("embedder")
        self.estimator = kwargs.get("estimator")
        # keep last fitted number of classes and samples
        self.n_classes = 0
        self.n_samples = 0

    @property
    def detector(self) -> str:
        """
        :return: detector model name
        :rtype: str
        """

        return self._detector.name

    @detector.setter
    def detector(self, name: Optional[str]) -> None:
        """Face detector model to use:
            - 'scrfd': insightface SCRFD model (default with insightface)
            - 'scrfd_antelopev2': insightface SCRFD model (antelopev2 pack)
            - 'hog': dlib "Histogram Oriented Gradients" model
                (default without insightface)
            - 'mmod': dlib "Max-Margin Object Detection" model

        :param name: detector model name
        :type name: str
        """

        Detector = _models.get(_resolve(name, _DETECTOR_DEFAULTS, "detector"))
        self._detector = Detector()

    @property
    def embedder(self) -> str:
        """
        :return: embedder model name
        :rtype: str
        """

        return self._embedder.name

    @embedder.setter
    def embedder(self, name: Optional[str]) -> None:
        """Face embedder model to use:
            - 'arcface': insightface ArcFace model
                (default with insightface)
            - 'arcface_antelopev2': insightface ArcFace model
                (antelopev2 pack)
            - 'resnet': dlib ResNet model (default without insightface)

        :param name: embedder model name
        :type name: str
        """

        Embedder = _models.get(_resolve(name, _EMBEDDER_DEFAULTS, "embedder"))
        self._embedder = Embedder()

    @property
    def estimator(self) -> str:
        """
        :return: estimator model name
        :rtype: str
        """

        return self._estimator.name

    @estimator.setter
    def estimator(self, name: Optional[str]) -> None:
        """Estimator model to use:
            - 'basic': linear comparing estimator (default)

        :param name: estimator model name
        :type name: str
        """

        Estimator = _models.get(_resolve(name, _ESTIMATOR_DEFAULTS, "estimator"))
        self._estimator = Estimator()

    def save(self, filename: str) -> None:
        """Save engine object state to the file as JSON.

        Persisting the object state as lightweight engine instance which
        contains only model name strings instead of model objects itself.
        Upon loading model objects will have to be re-initialized.

        Any extra instance attributes must be JSON-serializable.
        """

        # save estimator model's state only if there is something to persist
        if self.n_samples > 0:
            self._estimator.save(os.path.dirname(filename))

        # everything beyond the model trio and counters goes to "extra"
        extra = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in ("_detector", "_embedder", "_estimator", "n_classes", "n_samples")
        }
        data = {
            "format": "face-engine",
            "format_version": 1,
            "library_version": __version__,
            "detector": self.detector,
            "embedder": self.embedder,
            "estimator": self.estimator,
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
        }
        if extra:
            data["extra"] = extra
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def _fit(
        self, embeddings: np.ndarray, class_names: List[Any], **kwargs: Any
    ) -> "FaceEngine":
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

    def fit(
        self, images: List[str], class_names: List[Any], **kwargs: Any
    ) -> "FaceEngine":
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

        assert len(images) == len(
            class_names
        ), "the number of images and class_names has to be equal"

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

    def predict(self, embeddings: np.ndarray) -> Tuple[List[float], List[Any]]:
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

    def make_prediction(
        self, image: Union[str, bytes, os.PathLike, np.ndarray], **kwargs: Any
    ) -> Tuple[np.ndarray, List[Any]]:
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

        if not hasattr(image, "shape"):
            image = imread(image)

        bounding_boxes, extra = self.find_faces(image, **kwargs)
        embeddings = self.compute_embeddings(image, bounding_boxes, **extra)
        class_names = self.predict(embeddings)[1]
        return bounding_boxes, class_names

    def find_faces(
        self,
        image: Union[str, bytes, os.PathLike, np.ndarray],
        limit: Optional[int] = None,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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

        if not hasattr(image, "shape"):
            image = imread(image)

        # original image height and width
        height, width = image.shape[0:2]

        bbs, extra = self._detector.detect(image)

        n_det = len(bbs)
        if isinstance(limit, int) and 0 < limit < n_det:
            # keep the largest faces; stable sort preserves detector order
            # for equal-sized boxes
            areas = np.asarray([(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in bbs])
            indices = np.argsort(-areas, kind="stable")[:limit]
            bbs = np.asarray(bbs)[indices]
            extra = {key: np.asarray(value)[indices] for key, value in extra.items()}

        if normalize:
            bbs = bbs / ([width, height] * 2)
        return bbs, extra

    def compute_embeddings(
        self, image: np.ndarray, bounding_boxes: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
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

        embeddings = self._embedder.compute_embeddings(image, bounding_boxes, **kwargs)
        return embeddings
