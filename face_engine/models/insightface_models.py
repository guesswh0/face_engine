import os

import numpy as np
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from face_engine import RESOURCES
from face_engine.exceptions import FaceNotFoundError
from face_engine.models import Detector, Embedder
from face_engine.fetching import fetch_file

# download dependent models
fetch_file(
    "http://storage.insightface.ai/files/models/buffalo_l.zip",
    os.path.join(RESOURCES, 'models/buffalo_l')
)


class RetinaFaceDetector(Detector, name='retina_face'):

    def __init__(self):
        model = os.path.join(RESOURCES, 'models/buffalo_l/det_10g.onnx')
        self._detector = model_zoo.get_model(model)
        self._detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    def detect(self, image, limit=None):
        bbs, kpss = self._detector.detect(image)
        n_det = bbs.shape[0]
        if n_det < 1:
            raise FaceNotFoundError

        det_scores = bbs[:, 4]
        extra = dict(det_scores=det_scores, kpss=kpss)
        return bbs[:, :4], extra


class ArcFaceEmbedder(Embedder, name='arcface', dim=512):
    def __init__(self):
        model = os.path.join(RESOURCES, 'models/buffalo_l/w600k_r50.onnx')
        self._embedder = model_zoo.get_model(model)
        self._embedder.prepare(ctx_id=0)

    def compute_embeddings(self, image, bounding_boxes, **kwargs):
        assert 'kpss' in kwargs, (
            "kpss is not in kwargs, probably using wrong detector model")
        kpss = kwargs.get('kpss')

        embeddings = []
        for bb, kps in zip(bounding_boxes, kpss):
            aimg = face_align.norm_crop(image, kps)
            embedding = self._embedder.get_feat(aimg).flatten()
            embeddings.append(embedding / np.linalg.norm(embedding))
        return np.array(embeddings)
