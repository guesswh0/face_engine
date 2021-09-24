import logging
import unittest

import numpy as np

from face_engine import FaceEngine
from face_engine.exceptions import FaceNotFoundError, TrainError
from face_engine.models import Detector, Embedder, Estimator
from face_engine.tools import imread
from tests import TestCase, dlib


class TestFaceEngine(TestCase):

    def setUp(self):
        self.test_engine = FaceEngine()
        self.empty_engine = FaceEngine(
            detector='abstract_detector',
            embedder='abstract_embedder',
            estimator='abstract_estimator'
        )

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_init_default_models(self):
        self.assertEqual(self.test_engine.detector, 'hog')
        self.assertEqual(self.test_engine.embedder, 'resnet')
        self.assertEqual(self.test_engine.estimator, 'basic')

    @unittest.skipIf(dlib, "dlib package is installed")
    def test_init_default_models_without_dlib(self):
        self.assertEqual(self.test_engine.detector, 'abstract_detector')
        self.assertEqual(self.test_engine.embedder, 'abstract_embedder')
        self.assertEqual(self.test_engine.estimator, 'basic')

    def test_init_abstract_models(self):
        self.assertEqual(self.empty_engine.detector, 'abstract_detector')
        self.assertEqual(self.empty_engine.embedder, 'abstract_embedder')
        self.assertEqual(self.empty_engine.estimator, 'abstract_estimator')

    def test_explicit_init_with_not_existing_models(self):
        logging.disable(logging.WARNING)
        engine = FaceEngine(
            detector='my_detector',
            embedder='my_embedder',
            estimator='my_estimator'
        )
        self.assertEqual(engine.detector, 'abstract_detector')
        self.assertEqual(engine.embedder, 'abstract_embedder')
        self.assertEqual(engine.estimator, 'abstract_estimator')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_setters_with_defaults(self):
        self.empty_engine.detector = 'hog'
        self.empty_engine.embedder = 'resnet'
        self.empty_engine.estimator = 'basic'
        self.assertEqual(self.empty_engine.detector, 'hog')
        self.assertIsInstance(self.empty_engine._detector, Detector)
        self.assertEqual(self.empty_engine.embedder, 'resnet')
        self.assertIsInstance(self.empty_engine._embedder, Embedder)
        self.assertEqual(self.empty_engine.estimator, 'basic')
        self.assertIsInstance(self.empty_engine._estimator, Estimator)

    def test_setters_with_abstract_models(self):
        self.test_engine.detector = 'abstract_detector'
        self.test_engine.embedder = 'abstract_embedder'
        self.test_engine.estimator = 'abstract_estimator'
        self.assertIsInstance(self.test_engine._detector, Detector)
        self.assertEqual(self.test_engine.detector, 'abstract_detector')
        self.assertIsInstance(self.test_engine._embedder, Embedder)
        self.assertEqual(self.test_engine.embedder, 'abstract_embedder')
        self.assertIsInstance(self.test_engine._estimator, Estimator)
        self.assertEqual(self.test_engine.estimator, 'abstract_estimator')

    def test_setters_with_not_existing_models(self):
        logging.disable(logging.WARNING)
        self.empty_engine.detector = 'my_detector'
        self.empty_engine.embedder = 'my_embedder'
        self.empty_engine.estimator = 'my_estimator'
        self.assertNotEqual(self.empty_engine.detector, 'my_detector')
        self.assertEqual(self.empty_engine.detector, 'abstract_detector')
        self.assertNotEqual(self.empty_engine.embedder, 'my_embedder')
        self.assertEqual(self.empty_engine.embedder, 'abstract_embedder')
        self.assertNotEqual(self.empty_engine.estimator, 'my_estimator')
        self.assertEqual(self.empty_engine.estimator, 'abstract_estimator')

    @unittest.skipIf(dlib, "dlib package is installed")
    def test_setters_with_none_without_dlib(self):
        self.test_engine.detector = None
        self.test_engine.embedder = None
        self.test_engine.estimator = None
        self.assertEqual(self.test_engine.detector, 'abstract_detector')
        self.assertEqual(self.test_engine.embedder, 'abstract_embedder')
        self.assertEqual(self.test_engine.estimator, 'basic')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_setters_with_none_with_dlib(self):
        self.test_engine.detector = None
        self.test_engine.embedder = None
        self.test_engine.estimator = None
        self.assertEqual(self.test_engine.detector, 'hog')
        self.assertEqual(self.test_engine.embedder, 'resnet')
        self.assertEqual(self.test_engine.estimator, 'basic')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_getters(self):
        self.assertIsInstance(self.test_engine.detector, str)
        self.assertIsInstance(self.test_engine.embedder, str)
        self.assertIsInstance(self.test_engine.estimator, str)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_fit_bubbles(self):
        images = [self.bubbles1, self.bubbles2]
        classes = [0, 0]
        self.test_engine.fit(images, classes)
        self.assertEqual(self.test_engine.n_samples, 2)
        self.assertEqual(self.test_engine.n_classes, 1)

    def test_fit_raises_assertion_error(self):
        images = [self.bubbles1, self.family, self.drive]
        classes = [0, 1]
        with self.assertRaises(AssertionError):
            self.test_engine.fit(images, classes)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_predict_return_data_types(self):
        self.test_engine.fit([self.bubbles1], [0])
        image = imread(self.bubbles1)
        bbs, extra = self.test_engine.find_faces(image, limit=1)
        embeddings = self.test_engine.compute_embeddings(image, bbs, **extra)
        data = self.test_engine.predict(embeddings)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], list)
        self.assertIsInstance(data[1], list)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_predict_before_fit_raises_train_error(self):
        image = imread(self.bubbles1)
        bbs, extra = self.test_engine.find_faces(image)
        embeddings = self.test_engine.compute_embeddings(image, bbs, **extra)
        with self.assertRaises(TrainError):
            self.test_engine.predict(embeddings)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_return_data_types(self):
        self.test_engine.fit([self.bubbles1], [0])
        data = self.test_engine.make_prediction(self.bubbles2)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], list)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_before_fit_raises_train_error(self):
        with self.assertRaises(TrainError):
            self.test_engine.make_prediction(self.bubbles1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_before_fit_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.test_engine.make_prediction(self.book_stack)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_return_data_types(self):
        data = self.test_engine.find_faces(self.bubbles1)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], np.ndarray)
        self.assertIsInstance(data[1], dict)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_return_single_bounding_box(self):
        bbs, _ = self.test_engine.find_faces(self.family, limit=1)
        # returns single bounding box of 4 points
        self.assertEqual(len(bbs), 1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_image_content(self):
        content = imread(self.family)
        data = self.test_engine.find_faces(content)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_image_uri(self):
        data = self.test_engine.find_faces(self.bubbles1)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_normalize(self):
        bbs, _ = self.test_engine.find_faces(self.family, normalize=True)
        self.assertTrue(all(p <= 1.0 for p in bbs.flatten()))

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.test_engine.find_faces(self.book_stack)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_returns_multiple_bounding_boxes(self):
        bbs, _ = self.test_engine.find_faces(self.family)
        # family image has three faces
        self.assertGreater(len(bbs), 1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_compute_embeddings_vector_dimension(self):
        bubbles1 = imread(self.bubbles1)
        bbs, _ = self.test_engine.find_faces(bubbles1)
        embeddings = self.test_engine.compute_embeddings(bubbles1, bbs)
        self.assertEqual(
            embeddings.size, self.test_engine._embedder.embedding_dim)


if __name__ == '__main__':
    unittest.main()
