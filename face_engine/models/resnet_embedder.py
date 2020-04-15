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
from face_engine.models import Embedder


class ResNetEncoder(Embedder, name='resnet', dim=128):
    """Dlib pre-trained face recognition ResNet model.

        -   face alignment pre-processing used with 5 point shape_predictor.

    References:
        [1]  http://dlib.net/python/index.html

        [2] http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

        [3] http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
    """

    def __init__(self) -> None:
        try:
            self._face_encoder = dlib.face_recognition_model_v1(
                os.path.join(
                    RESOURCES,
                    "data/dlib_face_recognition_resnet_model_v1.dat"
                ))
            self._shape_predictor = dlib.shape_predictor(
                os.path.join(
                    RESOURCES,
                    "data/shape_predictor_5_face_landmarks.dat"
                ))
        except RuntimeError:
            logger.error(
                "Embedder model 'resnet' data files not found! "
                "Use `fetch_models` and try again."
            )
            raise

    def compute_embedding(self, image, bounding_box):
        bb = dlib.rectangle(bounding_box[0], bounding_box[1],
                            bounding_box[2], bounding_box[3])
        shape = self._shape_predictor(image, bb)

        # Aligned to shape and cropped by bounding box face image
        # default shape (150, 150, 3)
        face_image = dlib.get_face_chip(image, shape)

        embedding = self._face_encoder.compute_face_descriptor(face_image)
        return np.array(embedding)

    def compute_embeddings(self, image, bounding_boxes):
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
