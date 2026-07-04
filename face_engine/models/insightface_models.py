import os

import numpy as np
import onnxruntime
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

from face_engine import RESOURCES
from face_engine.exceptions import FaceNotFoundError
from face_engine.fetching import fetch_file
from face_engine.models import Detector, Embedder

# model packs from the insightface v0.7 release assets
# (storage.insightface.ai is no longer available);
# pre-trained weights are for non-commercial research purposes only
_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{}.zip"


def _providers():
    """Cuda when available, cpu otherwise.

    Insightface requests CUDAExecutionProvider unconditionally, making
    onnxruntime warn on every non-CUDA machine.
    """
    available = onnxruntime.get_available_providers()
    return [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ]


def _fetch_pack(pack, filename):
    """Fetch insightface model pack and return path to the model file."""
    extract_dir = os.path.join(RESOURCES, "models", pack)
    fetch_file(_PACK_URL.format(pack), extract_dir)
    # some packs (antelopev2) extract into a nested directory
    for path in (
        os.path.join(extract_dir, filename),
        os.path.join(extract_dir, pack, filename),
    ):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{filename} not found in {pack} model pack")


class SCRFDDetector(Detector, name="scrfd", aliases=("retina_face",)):
    """InsightFace SCRFD-10GF face detector from the ``buffalo_l`` model pack.

    .. note::
        * registered as ``scrfd``; the pre-3.0 name ``retina_face`` is kept
          as a deprecated alias.
        * model weights are licensed for non-commercial research
          purposes only.

    References:
        1. https://github.com/deepinsight/insightface
        2. SCRFD: https://arxiv.org/abs/2105.04714
    """

    pack = "buffalo_l"
    file = "det_10g.onnx"

    def __init__(self):
        model = _fetch_pack(self.pack, self.file)
        self._detector = model_zoo.get_model(model, providers=_providers())
        self._detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    def detect(self, image):
        bbs, kpss = self._detector.detect(image)
        n_det = bbs.shape[0]
        if n_det < 1:
            raise FaceNotFoundError

        det_scores = bbs[:, 4]
        extra = dict(det_scores=det_scores, kpss=kpss)
        return bbs[:, :4], extra


class SCRFDAntelopeV2Detector(SCRFDDetector, name="scrfd_antelopev2"):
    """InsightFace SCRFD-10GF (bnkps) detector from the ``antelopev2``
    model pack.

    .. note::
        * model weights are licensed for non-commercial research
          purposes only.
    """

    pack = "antelopev2"
    file = "scrfd_10g_bnkps.onnx"


class ArcFaceEmbedder(Embedder, name="arcface", dim=512):
    """InsightFace ArcFace embedder (ResNet50@WebFace600K) from the
    ``buffalo_l`` model pack.

    .. note::
        * requires ``kpss`` face keypoints from a SCRFD detector for
          face alignment.
        * model weights are licensed for non-commercial research
          purposes only.

    References:
        1. https://github.com/deepinsight/insightface
        2. ArcFace: https://arxiv.org/abs/1801.07698
    """

    pack = "buffalo_l"
    file = "w600k_r50.onnx"

    def __init__(self):
        model = _fetch_pack(self.pack, self.file)
        self._embedder = model_zoo.get_model(model, providers=_providers())
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


class ArcFaceAntelopeV2Embedder(ArcFaceEmbedder, name="arcface_antelopev2", dim=512):
    """InsightFace ArcFace embedder (ResNet100@Glint360K) from the
    ``antelopev2`` model pack — the strongest insightface pack.

    .. note::
        * model weights are licensed for non-commercial research
          purposes only.
    """

    pack = "antelopev2"
    file = "glintr100.onnx"
