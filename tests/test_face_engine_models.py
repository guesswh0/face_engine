import os
import unittest

import numpy as np

from extra.dataset import load_dataset
from face_engine.exceptions import FaceNotFoundError
from face_engine.models import Detector, Embedder, Estimator, _models
from face_engine.models.basic_estimator import BasicEstimator
from face_engine.tools import imread
from tests import TestCase, dlib, insightface, onnxruntime

if dlib:
    from face_engine.models.dlib_models import HOGDetector, MMODDetector, ResNetEmbedder
if insightface:
    from face_engine.models.insightface_models import (
        ArcFaceAntelopeV2Embedder,
        ArcFaceEmbedder,
        SCRFDAntelopeV2Detector,
        SCRFDDetector,
    )

# antelopev2 tests are gated behind an env var (407 MB model pack download)
ANTELOPE = os.environ.get("FACE_ENGINE_TEST_ANTELOPE") == "1"


class TestDetector(TestCase):
    """Test cases to test detectors.

    To use this tests @override setUp() method with your detector

    Note that this abstract class is not collected by unittest.
    """

    def setUp(self):
        self.detector = Detector()

    def test_detect_return_data_is_tuple(self):
        data = self.detector.detect(imread(self.bubbles1))
        self.assertIsNotNone(data)

    def test_detect_return_bbs_type(self):
        bbs, _ = self.detector.detect(imread(self.bubbles1))
        self.assertIsInstance(bbs, np.ndarray)

    def test_detect_return_multiple_bbs(self):
        bbs, _ = self.detector.detect(imread(self.family))
        self.assertGreater(len(bbs), 1)

    def test_detect_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.detector.detect(imread(self.book_stack))


@unittest.skipUnless(dlib, "dlib package is not installed")
class TestHOGDetector(TestDetector):

    def setUp(self):
        self.detector = HOGDetector()


@unittest.skipUnless(dlib, "dlib package is not installed")
class TestMMODDetector(TestDetector):

    def setUp(self):
        self.detector = MMODDetector()


@unittest.skipUnless(insightface, "insightface package is not installed")
class TestSCRFDDetector(TestDetector):

    def setUp(self):
        self.detector = SCRFDDetector()


@unittest.skipUnless(insightface and ANTELOPE, "antelopev2 tests are not enabled")
class TestSCRFDAntelopeV2Detector(TestDetector):

    def setUp(self):
        self.detector = SCRFDAntelopeV2Detector()


class TestEmbedder(TestCase):
    """Test cases to test embedders.

    To use this tests @override setUp() method with your embedder

    Note that this abstract class is not collected by unittest.
    """

    def setUp(self):
        self.image = imread(self.bubbles1)
        self.bbs = np.array([[278, 132, 618, 471]])
        self.embedder = Embedder()

    def test_compute_embedding_return_data_is_numpy_array(self):
        data = self.embedder.compute_embeddings(self.image, self.bbs)
        self.assertIsInstance(data, np.ndarray)

    def test_compute_embedding_return_data_shape(self):
        data = self.embedder.compute_embeddings(self.image, self.bbs)
        self.assertEqual(data.shape, (1, self.embedder.embedding_dim))

    def test_compute_embeddings_return_data_is_numpy_array(self):
        data = self.embedder.compute_embeddings(self.image, self.bbs)
        self.assertIsInstance(data, np.ndarray)

    def test_compute_embeddings_return_data_shape(self):
        data = self.embedder.compute_embeddings(self.image, self.bbs)
        self.assertEqual(data.shape, (1, self.embedder.embedding_dim))


@unittest.skipUnless(dlib, "dlib package is not installed")
class TestResNetEmbedder(TestEmbedder):

    def setUp(self):
        super().setUp()
        self.embedder = ResNetEmbedder()


class TestInsightFaceEmbedder(TestCase):
    """Test cases for insightface embedders which require detector
    keypoints (``kpss``) for face alignment.

    Note that this abstract class is not collected by unittest.
    """

    def setUp(self):
        self.image = imread(self.bubbles1)
        self.detector = None
        self.embedder = None

    def test_compute_embeddings_return_data_shape(self):
        bbs, extra = self.detector.detect(self.image)
        embeddings = self.embedder.compute_embeddings(self.image, bbs, **extra)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (len(bbs), self.embedder.embedding_dim))

    def test_compute_embeddings_without_kpss_raises(self):
        bbs, _ = self.detector.detect(self.image)
        with self.assertRaises(AssertionError):
            self.embedder.compute_embeddings(self.image, bbs)


@unittest.skipUnless(insightface, "insightface package is not installed")
class TestArcFaceEmbedder(TestInsightFaceEmbedder):

    def setUp(self):
        super().setUp()
        self.detector = SCRFDDetector()
        self.embedder = ArcFaceEmbedder()


@unittest.skipUnless(insightface and ANTELOPE, "antelopev2 tests are not enabled")
class TestArcFaceAntelopeV2Embedder(TestInsightFaceEmbedder):

    def setUp(self):
        super().setUp()
        self.detector = SCRFDAntelopeV2Detector()
        self.embedder = ArcFaceAntelopeV2Embedder()


@unittest.skipUnless(onnxruntime, "onnxruntime package is not installed")
class TestMiniFASNetAntispoof(TestCase):

    def setUp(self):
        from face_engine.models.minifasnet import MiniFASNetAntispoof

        self.antispoof = MiniFASNetAntispoof()
        self.image = imread(self.bubbles1)
        self.bbs = np.array([[278, 132, 618, 471]])

    def test_registered_under_canonical_name(self):
        self.assertIs(_models["minifasnet"], type(self.antispoof))

    def test_predict_return_data_shape_and_range(self):
        scores = self.antispoof.predict(self.image, self.bbs)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(0.0 <= scores[0] <= 1.0)

    def test_predict_live_photo_scores_high(self):
        scores = self.antispoof.predict(self.image, self.bbs)
        self.assertGreater(scores[0], 0.5)

    def test_predict_multiple_faces(self):
        image = imread(self.family)
        bbs = np.array(
            [[100, 100, 200, 220], [300, 120, 400, 240], [500, 110, 600, 230]]
        )
        scores = self.antispoof.predict(image, bbs)
        self.assertEqual(scores.shape, (3,))

    def test_predict_empty_bounding_boxes(self):
        scores = self.antispoof.predict(self.image, np.empty((0, 4)))
        self.assertEqual(scores.shape, (0,))


class TestEstimator(TestCase):
    """Test cases to test estimators.

    To use this tests @override setUp() method with your estimator

    Note that this abstract class is not collected by unittest.
    """

    def setUp(self):
        self.estimator = Estimator()
        train_dataset = load_dataset(self.train)
        self.train_emb = train_dataset[3]
        self.train_names = train_dataset[1]
        test_dataset = load_dataset(self.test)
        self.test_emb = test_dataset[3]
        self.test_names = test_dataset[1]

    def test_fit_with_defaults(self):
        self.estimator.fit(np.array(self.train_emb), self.train_names)
        # fit completes without troubles
        self.assertTrue(True)

    def test_predict_test_dataset(self):
        self.estimator.fit(np.array(self.train_emb), self.train_names)
        names = self.estimator.predict(np.array(self.test_emb))[1]
        self.assertEqual(self.test_names, names)


class TestBasicEstimator(TestEstimator):

    def setUp(self):
        super().setUp()
        self.estimator = BasicEstimator()


# remove abstract tests
del TestDetector
del TestEmbedder
del TestInsightFaceEmbedder
del TestEstimator

if __name__ == "__main__":
    unittest.main()
