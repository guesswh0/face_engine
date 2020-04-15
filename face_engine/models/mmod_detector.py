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

import dlib
import numpy as np

from face_engine import logger, RESOURCES
from face_engine.exceptions import FaceError
from face_engine.models import Detector


class MMODDetector(Detector, name='mmod'):
    """Dlib pre-trained CNN model with "Max-Margin Object Detection"
    loss function.

        -   to run in realtime requires high-end Nvidia GPU and CUDA
            with cuDNN lib installed.
        -   compounds equal-sized (width and height) bounding boxes
            for all detections.

    References:
        [1]  http://dlib.net/python/index.html

        [2] http://dlib.net/files/mmod_human_face_detector.dat.bz2

        [3] https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
    """

    def __init__(self) -> None:
        try:
            self._cnn_face_detector = dlib.cnn_face_detection_model_v1(
                os.path.join(RESOURCES, "data/mmod_human_face_detector.dat"))
        except RuntimeError:
            logger.error(
                "Detector model 'mmod' data files not found! "
                "Use `fetch_models` and try again."
            )
            raise

    def detect_all(self, image):
        detections = self._cnn_face_detector(image)
        n_det = len(detections)
        if n_det < 1:
            raise FaceError()

        img_size = np.asarray(image.shape)[0:2]
        bounding_boxes = np.empty(shape=(n_det, 4), dtype=np.uint16)
        confidence_scores = np.empty(n_det, dtype=np.float32)
        for i, det in enumerate(detections):
            bounding_boxes[i, 0] = np.maximum(det.rect.left(), 0)
            bounding_boxes[i, 1] = np.maximum(det.rect.top(), 0)
            bounding_boxes[i, 2] = np.minimum(det.rect.right(), img_size[1])
            bounding_boxes[i, 3] = np.minimum(det.rect.bottom(), img_size[0])
            confidence_scores[i] = det.confidence
        return confidence_scores, bounding_boxes

    def detect_one(self, image):
        confidences, bounding_boxes = self.detect_all(image)
        # dlib bounding boxes are all equal sized
        # returning first face bounding_box
        return confidences[0], bounding_boxes[0]
