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

import dlib
import numpy as np

from face_engine.exceptions import FaceError
from face_engine.models import Detector


class HOGDetector(Detector, name='hog'):
    """Dlib pre-trained "Histogram Oriented Gradients" model.

        -   detector does not give a confidence scores for detections.
        -   compounds equal-sized (width and height) bounding boxes
            for all detections.
    """

    def __init__(self):
        self._face_detector = dlib.get_frontal_face_detector()

    def detect_all(self, image):
        detections = self._face_detector(image)
        n_det = len(detections)
        if n_det < 1:
            raise FaceError()

        img_size = np.asarray(image.shape)[0:2]
        bounding_boxes = np.empty(shape=(n_det, 4), dtype=np.uint16)
        for i, rect in enumerate(detections):
            bounding_boxes[i, 0] = np.maximum(rect.left(), 0)
            bounding_boxes[i, 1] = np.maximum(rect.top(), 0)
            bounding_boxes[i, 2] = np.minimum(rect.right(), img_size[1])
            bounding_boxes[i, 3] = np.minimum(rect.bottom(), img_size[0])
        return None, bounding_boxes

    def detect_one(self, image):
        _, bounding_boxes = self.detect_all(image)
        # dlib bounding boxes are all equal sized
        # returning first face bounding_box
        return None, bounding_boxes[0]
