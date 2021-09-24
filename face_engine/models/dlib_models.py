import os

import dlib
import numpy as np

from face_engine import RESOURCES
from face_engine.exceptions import FaceNotFoundError
from face_engine.fetching import fetch_file
from face_engine.models import Detector, Embedder

# download dependent models
for url in [
    "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
    "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
]:
    fetch_file(url, os.path.join(RESOURCES, 'models/dlib'))


class HOGDetector(Detector, name='hog'):
    """Dlib "Histogram Oriented Gradients" model.

    .. note::
        * bounding box sizes are equal for all detections.
        * detector does not provide confidence scores for detections.
    """

    def __init__(self):
        self._face_detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        detections = self._face_detector(image)
        n_det = len(detections)
        if n_det < 1:
            raise FaceNotFoundError

        height, width = image.shape[0:2]
        bounding_boxes = np.array([
            [
                max(rect.left(), 0),
                max(rect.top(), 0),
                min(rect.right(), width),
                min(rect.bottom(), height)
            ]
            for rect in detections])
        return bounding_boxes, dict()


class MMODDetector(Detector, name='mmod'):
    """Dlib pre-trained CNN model with "Max-Margin Object Detection"
    loss function.

    .. note::
        * bounding box sizes are equal for all detections.
        * to run in realtime requires high-end Nvidia GPU with CUDA/cuDNN.

    References:
        1. http://dlib.net/python/index.html
        2. https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
        3. http://dlib.net/files/mmod_human_face_detector.dat.bz2
    """

    def __init__(self) -> None:
        self._cnn_face_detector = dlib.cnn_face_detection_model_v1(
            os.path.join(RESOURCES, "models/dlib/mmod_human_face_detector.dat"))

    def detect(self, image):
        detections = self._cnn_face_detector(image)
        n_det = len(detections)
        if n_det < 1:
            raise FaceNotFoundError

        height, width = image.shape[:2]
        det_scores = list()
        bounding_boxes = list()
        for det in detections:
            bounding_boxes.append(
                [
                    max(det.rect.left(), 0),
                    max(det.rect.top(), 0),
                    min(det.rect.right(), width),
                    min(det.rect.bottom(), height)
                ])
            det_scores.append(det.confidence)
        bounding_boxes = np.array(bounding_boxes)
        extra = dict(det_scores=det_scores)
        return bounding_boxes, extra


class ResNetEmbedder(Embedder, name='resnet', dim=128):
    """ Dlib pre-trained face recognition ResNet model.

    .. note::
        * face alignment pre-processing used with 5 point shape_predictor.


    References:
        1. http://dlib.net/python/index.html
        2. http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
        3. http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    """

    def __init__(self) -> None:
        self._face_encoder = dlib.face_recognition_model_v1(
            os.path.join(RESOURCES, "models/dlib/dlib_face_recognition_resnet_model_v1.dat"))
        self._shape_predictor = dlib.shape_predictor(
            os.path.join(RESOURCES, "models/dlib/shape_predictor_5_face_landmarks.dat"))

    def compute_embeddings(self, image, bounding_boxes, **kwargs):
        shapes = dlib.full_object_detections()
        for bounding_box in bounding_boxes:
            bb = dlib.rectangle(bounding_box[0], bounding_box[1],
                                bounding_box[2], bounding_box[3])
            shapes.append(self._shape_predictor(image, bb))

        # Aligned to shape and cropped by bounding boxes face images
        # default shape (n_faces, 150, 150, 3)
        face_images = dlib.get_face_chips(image, shapes)

        embeddings = self._face_encoder.compute_face_descriptor(face_images)
        return np.array(embeddings)
