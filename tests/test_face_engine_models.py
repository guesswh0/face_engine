import unittest

import numpy as np

from extra.dataset import load_dataset
from face_engine.exceptions import FaceNotFoundError
from face_engine.models import Detector, Embedder, Estimator
from face_engine.models.basic_estimator import BasicEstimator
from face_engine.tools import imread
from tests import TestCase, dlib

if dlib:
    from face_engine.models.dlib_models import (
        HOGDetector, MMODDetector, ResNetEmbedder)


class TestDetector(TestCase):
    """Test cases to test detectors.

    To use this tests @override setUp() method with your detector

    Note that this abstract class is not collected by unittest.
    """

    def setUp(self):
        self.detector = Detector()

    def test_detect_one_return_data_is_tuple(self):
        data = self.detector.detect_one(imread(self.bubbles1))
        self.assertIsNotNone(data)

    def test_detect_one_return_bb_type(self):
        bbs, _ = self.detector.detect_one(imread(self.bubbles1))
        self.assertIsInstance(bbs, np.ndarray)

    def test_detect_one_return_one_bb(self):
        bbs, _ = self.detector.detect_one(imread(self.bubbles1))
        self.assertEqual(len(bbs), 1)

    def test_detect_one_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.detector.detect_one(imread(self.cat))

    def test_detect_all_return_data_is_tuple(self):
        data = self.detector.detect_all(imread(self.bubbles1))
        self.assertIsNotNone(data)

    def test_detect_all_return_bbs_type(self):
        bbs, _ = self.detector.detect_all(imread(self.bubbles1))
        self.assertIsInstance(bbs, np.ndarray)

    def test_detect_all_return_multiple_bbs(self):
        bbs, _ = self.detector.detect_all(imread(self.family))
        self.assertGreater(len(bbs), 1)

    def test_detect_all_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.detector.detect_all(imread(self.cat))


@unittest.skipUnless(dlib, "dlib package is not installed")
class TestHOGDetector(TestDetector):

    def setUp(self):
        self.detector = HOGDetector()


@unittest.skipUnless(dlib, "dlib package is not installed")
class TestMMODDetector(TestDetector):

    def setUp(self):
        self.detector = MMODDetector()


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
del TestEstimator

if __name__ == '__main__':
    unittest.main()
