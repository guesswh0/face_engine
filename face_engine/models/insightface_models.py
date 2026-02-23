import os

import numpy as np
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from face_engine import RESOURCES
from face_engine.exceptions import FaceNotFoundError
from face_engine.models import Detector, Embedder
from face_engine.fetching import fetch_file

BUFFALO_L_URL = "http://storage.insightface.ai/files/models/buffalo_l.zip"
ANTELOPE_V2_URL = "https://huggingface.co/vladmandic/insightface-faceanalysis/resolve/main/antelopev2.zip"


class InsightFaceDetector(Detector):
    def __init__(self, model_pack_url, model_pack_name, model_file, det_input_size=(640, 640), det_thresh=0.5):
        self._model_pack_url = model_pack_url
        self._model_pack_name = model_pack_name
        self._model_file = model_file

        extract_dir = os.path.join(RESOURCES, f"models/{self._model_pack_name}")
        fetch_file(self._model_pack_url, extract_dir)

        # Check if the file is directly in extract_dir or nested
        model_path = os.path.join(extract_dir, self._model_file)
        # Note: In restricted environments where files cannot be downloaded, os.path.exists might fail.
        # We try to detect the path structure if files exist.
        if not os.path.exists(model_path):
             nested_path = os.path.join(extract_dir, self._model_pack_name, self._model_file)
             if os.path.exists(nested_path):
                 model_path = nested_path

        self._detector = model_zoo.get_model(model_path)
        self._detector.prepare(ctx_id=0, input_size=det_input_size, det_thresh=det_thresh)

    def detect(self, image, limit=None):
        bbs, kpss = self._detector.detect(image)
        if bbs is None:
            raise FaceNotFoundError

        n_det = bbs.shape[0]
        if n_det < 1:
            raise FaceNotFoundError

        det_scores = bbs[:, 4]
        extra = dict(det_scores=det_scores, kpss=kpss)
        return bbs[:, :4], extra


class RetinaFaceDetector(InsightFaceDetector, name="retina_face"):

    def __init__(self):
        super().__init__(
            model_pack_url=BUFFALO_L_URL,
            model_pack_name="buffalo_l",
            model_file="det_10g.onnx"
        )


class SCRFDDetector(InsightFaceDetector, name="scrfd"):

    def __init__(self):
        super().__init__(
            model_pack_url=ANTELOPE_V2_URL,
            model_pack_name="antelopev2",
            model_file="scrfd_10g_bnkps.onnx"
        )


class InsightFaceEmbedder(Embedder):
    def __init__(self, model_pack_url, model_pack_name, model_file):
        self._model_pack_url = model_pack_url
        self._model_pack_name = model_pack_name
        self._model_file = model_file

        extract_dir = os.path.join(RESOURCES, f"models/{self._model_pack_name}")
        fetch_file(self._model_pack_url, extract_dir)

        model_path = os.path.join(extract_dir, self._model_file)
        if not os.path.exists(model_path):
             nested_path = os.path.join(extract_dir, self._model_pack_name, self._model_file)
             if os.path.exists(nested_path):
                 model_path = nested_path

        self._embedder = model_zoo.get_model(model_path)
        self._embedder.prepare(ctx_id=0)

    def compute_embeddings(self, image, bounding_boxes, **kwargs):
        assert (
            "kpss" in kwargs
        ), "kpss is not in kwargs, probably using wrong detector model"
        kpss = kwargs.get("kpss")

        embeddings = []
        for bb, kps in zip(bounding_boxes, kpss):
            aimg = face_align.norm_crop(image, kps)
            embedding = self._embedder.get_feat(aimg).flatten()
            embeddings.append(embedding / np.linalg.norm(embedding))
        return np.array(embeddings)


class ArcFaceEmbedder(InsightFaceEmbedder, name="arcface", dim=512):
    def __init__(self):
        super().__init__(
            model_pack_url=BUFFALO_L_URL,
            model_pack_name="buffalo_l",
            model_file="w600k_r50.onnx"
        )


class ArcFaceResNet100Embedder(InsightFaceEmbedder, name="arcface_r100", dim=512):
    def __init__(self):
        super().__init__(
            model_pack_url=ANTELOPE_V2_URL,
            model_pack_name="antelopev2",
            model_file="glintr100.onnx"
        )
