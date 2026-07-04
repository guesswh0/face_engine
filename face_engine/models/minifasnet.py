import os

import numpy as np
import onnxruntime

from face_engine import RESOURCES
from face_engine.fetching import fetch_file
from face_engine.models import Antispoof

# onnx exports of the released minivision-ai/Silent-Face-Anti-Spoofing
# checkpoints (Apache-2.0), reproducible with extra/export_minifasnet.py;
# hosted as face-engine github release assets
_ASSET_URL = "https://github.com/guesswh0/face_engine/releases/download/v3.0.0/{}"


def _providers():
    """Cuda when available, cpu otherwise."""
    available = onnxruntime.get_available_providers()
    return [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ]


def _softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _resize_bilinear(image, out_h, out_w):
    """cv2.resize(INTER_LINEAR) equivalent, to avoid an opencv dependency."""
    in_h, in_w = image.shape[:2]
    ys = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    y0c, y1c = np.clip(y0, 0, in_h - 1), np.clip(y0 + 1, 0, in_h - 1)
    x0c, x1c = np.clip(x0, 0, in_w - 1), np.clip(x0 + 1, 0, in_w - 1)
    image = image.astype(np.float32)
    top = image[y0c][:, x0c] * (1 - wx) + image[y0c][:, x1c] * wx
    bottom = image[y1c][:, x0c] * (1 - wx) + image[y1c][:, x1c] * wx
    # official pipeline resizes uint8 images, so round like cv2 does
    return np.rint(top * (1 - wy) + bottom * wy)


class MiniFASNetAntispoof(Antispoof, name="minifasnet"):
    """MiniFASNet passive anti-spoofing ensemble (Silent-Face-Anti-Spoofing).

    Ensemble of the two released 80x80 models, each run on its own
    scale-cropped patch around the face; the averaged softmax gives the
    live-face probability.

    .. note::
        * both code and model weights are Apache-2.0 — usable in
          commercial products.
        * effective against printed photos and basic screen replays;
          not a certified (ISO/IEC 30107-3) liveness solution.

    References:
        1. https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    """

    # filename -> official crop scale (each model has its own)
    files = (
        ("2.7_80x80_MiniFASNetV2.onnx", 2.7),
        ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0),
    )
    input_size = (80, 80)

    def __init__(self):
        extract_dir = os.path.join(RESOURCES, "models", "minifasnet")
        self._sessions = []
        for filename, scale in self.files:
            fetch_file(_ASSET_URL.format(filename), extract_dir)
            session = onnxruntime.InferenceSession(
                os.path.join(extract_dir, filename), providers=_providers()
            )
            self._sessions.append((session, scale))

    def predict(self, image, bounding_boxes):
        n_faces = len(bounding_boxes)
        if n_faces == 0:
            return np.empty(0, dtype=np.float32)

        # models are trained on BGR images
        image = image[..., ::-1]
        probabilities = np.zeros((n_faces, 3), dtype=np.float32)
        for session, scale in self._sessions:
            batch = np.stack([self._patch(image, bb, scale) for bb in bounding_boxes])
            logits = session.run(None, {"input": batch})[0]
            probabilities += _softmax(logits)
        probabilities /= len(self._sessions)
        # softmax classes are (fake, live, fake)
        return probabilities[:, 1]

    def _patch(self, image, bounding_box, scale):
        """Scale-cropped face patch as float32 CHW tensor in [0, 255].

        Reproduces the official CropImage box math: aspect-preserving
        scaled box around the face, shifted back inside the image, then
        resized to the model input size (no normalization by design —
        the models were trained on raw 0-255 inputs).
        """
        src_h, src_w = image.shape[:2]
        left, upper, right, lower = bounding_box
        box_w, box_h = right - left + 1, lower - upper + 1

        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
        new_w, new_h = box_w * scale, box_h * scale
        center_x, center_y = left + box_w / 2, upper + box_h / 2

        left_top_x = center_x - new_w / 2
        left_top_y = center_y - new_h / 2
        right_bottom_x = center_x + new_w / 2
        right_bottom_y = center_y + new_h / 2
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        patch = image[
            int(left_top_y) : int(right_bottom_y) + 1,
            int(left_top_x) : int(right_bottom_x) + 1,
        ]
        patch = _resize_bilinear(patch, *self.input_size)
        return patch.transpose((2, 0, 1)).astype(np.float32)
