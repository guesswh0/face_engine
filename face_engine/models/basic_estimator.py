import json
import os

import numpy as np

from face_engine.exceptions import TrainError
from face_engine.models import Estimator


class BasicEstimator(Estimator, name="basic"):
    """Basic estimator model makes predictions by linear comparing each source
    embedding vector with each fitted embedding vectors.

    Model state is persisted as ``'basic.estimator.npz'`` (embeddings) and
    ``'basic.estimator.json'`` (class names) files.
    """

    def __init__(self):
        self.embeddings = None
        self.class_names = None

    def fit(self, embeddings, class_names, **kwargs):
        self.embeddings = embeddings
        self.class_names = class_names

    def predict(self, embeddings):
        if self.class_names is None:
            raise TrainError("Model is not fitted yet!")

        scores = []
        class_names = []
        for embedding in embeddings:
            distances = np.linalg.norm(self.embeddings - embedding, axis=1)
            index = np.argmin(distances)
            score = np.exp(-0.5 * distances[index] ** 2)
            scores.append(score)
            class_names.append(self.class_names[index])
        return scores, class_names

    def save(self, dirname):
        np.savez(
            os.path.join(dirname, "%s.estimator.npz" % self.name),
            embeddings=np.asarray(self.embeddings),
        )
        with open(
            os.path.join(dirname, "%s.estimator.json" % self.name),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"class_names": list(self.class_names)}, file)

    def load(self, dirname):
        npz = os.path.join(dirname, "%s.estimator.npz" % self.name)
        if not os.path.isfile(npz):
            legacy = os.path.join(dirname, "%s.estimator.p" % self.name)
            if os.path.isfile(legacy):
                raise RuntimeError(
                    "%s appears to be a legacy pickle produced by "
                    "face-engine < 3.0. Pickle persistence was removed in "
                    "3.0.0 for security. Re-fit the engine, then save() "
                    "again." % legacy
                )
        with np.load(npz, allow_pickle=False) as data:
            self.embeddings = data["embeddings"]
        with open(
            os.path.join(dirname, "%s.estimator.json" % self.name), encoding="utf-8"
        ) as file:
            self.class_names = json.load(file)["class_names"]
