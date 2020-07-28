import logging
import unittest

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

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_fit_raises_train_error(self):
        self.test_engine.limit = 1
        images = [self.bubbles1, self.bubbles2]
        classes = [0, 0]
        with self.assertRaises(TrainError):
            self.test_engine.fit(images, classes)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_fit_with_bounding_boxes(self):
        images = [self.bubbles1, self.bubbles2, self.drive]
        classes = [0, 0, 1]
        bbs = [(278, 132, 618, 471), (270, 75, 406, 211), (205, 157, 440, 393)]
        self.test_engine.fit(images, classes, bbs)
        self.assertEqual(self.test_engine.n_samples, 3)
        self.assertEqual(self.test_engine.n_classes, 2)

    def test_fit_raises_assertion_error(self):
        images = [self.bubbles1, self.family, self.drive]
        classes = [0, 1]
        with self.assertRaises(AssertionError):
            self.test_engine.fit(images, classes)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_predict_return_data_types(self):
        self.test_engine.fit([self.bubbles1], [0], [(278, 132, 618, 471)])
        image = imread(self.bubbles1)
        bb = self.test_engine.find_face(image)[1]
        embeddings = self.test_engine.compute_embedding(image, bb)
        data = self.test_engine.predict(embeddings)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], list)
        self.assertIsInstance(data[1], list)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_predict_before_fit_raises_train_error(self):
        image = imread(self.bubbles1)
        bb = self.test_engine.find_face(image)[1]
        embeddings = self.test_engine.compute_embedding(image, bb)
        with self.assertRaises(TrainError):
            self.test_engine.predict(embeddings)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_return_data_types(self):
        self.test_engine.fit([self.bubbles1], [0], [(278, 132, 618, 471)])
        data = self.test_engine.make_prediction(self.bubbles2)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsInstance(data[0], list)
        self.assertIsInstance(data[1], list)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_before_fit_raises_train_error(self):
        with self.assertRaises(TrainError):
            self.test_engine.make_prediction(self.bubbles1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_make_prediction_before_fit_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.test_engine.make_prediction(self.cat)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_return_data_types(self):
        data = self.test_engine.find_face(self.bubbles1)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsNone(data[0], None)
        self.assertIsInstance(data[1], tuple)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_return_single_bounding_box(self):
        bb = self.test_engine.find_face(self.bubbles1)[1]
        # returns single bounding box of 4 points
        self.assertEqual(len(bb), 4)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_with_image_content(self):
        content = imread(self.family)
        data = self.test_engine.find_face(content)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_with_image_uri(self):
        data = self.test_engine.find_face(self.bubbles1)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_with_normalize(self):
        bb = self.test_engine.find_face(self.family, normalize=True)[1]
        self.assertTrue(all(p <= 1.0 for p in bb))

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.test_engine.find_face(self.cat)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_return_data_types(self):
        data = self.test_engine.find_faces(self.family)
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data, tuple)
        self.assertIsNone(data[0], None)
        self.assertIsInstance(data[1], list)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_returns_multiple_bounding_boxes(self):
        bbs = self.test_engine.find_faces(self.family)[1]
        # family image has three faces
        self.assertEqual(len(bbs), 3)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_image_content(self):
        content = imread(self.family)
        data = self.test_engine.find_faces(content)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_image_uri(self):
        data = self.test_engine.find_faces(self.family)
        self.assertIsNotNone(data)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_roi(self):
        image = imread(self.bubbles1)
        h, w = image.shape[:2]
        # face is in the first half (by height)
        with self.subTest(roi='first_half'):
            bbs = self.test_engine.find_faces(image, roi=(0, 0, w, h // 2))[1]
            self.assertEqual(len(bbs), 1)
        # no face in the second half (by height)
        with self.subTest(roi='second_half'):
            with self.assertRaises(FaceNotFoundError):
                self.test_engine.find_faces(image, roi=(0, h // 2, w, h))

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_with_normalize(self):
        bbs = self.test_engine.find_faces(self.family, normalize=True)[1]
        self.assertTrue(all(all(p <= 1.0 for p in bb) for bb in bbs))

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_raises_face_not_found_error(self):
        with self.assertRaises(FaceNotFoundError):
            self.test_engine.find_faces(self.cat)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_params_by_iou(self):
        """Test find faces with params scale / roi / scale with roi
        returning bounding boxes by comparing them to original
        bounding box by IoU (Intersection over Union) value.

        Reference:
            https://en.wikipedia.org/wiki/Jaccard_index
        """

        image = imread(self.bubbles1)
        origin = self.test_engine.find_faces(image)[1][0]
        origin_area = (origin[2] - origin[0] + 1) * (origin[3] - origin[1] + 1)

        def get_iou(bb):
            area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
            left = max(origin[0], bb[0])
            upper = max(origin[1], bb[1])
            right = min(origin[2], bb[2])
            lower = min(origin[3], bb[3])
            intersection = max(0, right - left + 1) * max(0, lower - upper + 1)
            return intersection / float(origin_area + area - intersection)

        # freeze params to hog detector
        scale = 0.7
        h, w = image.shape[:2]
        roi = (0, 0, w - 200, h // 2)
        # only scale param
        bb1 = self.test_engine.find_faces(image, scale=scale)[1][0]
        # only roi param
        bb2 = self.test_engine.find_faces(image, roi=roi)[1][0]
        # scale with roi param
        bb3 = self.test_engine.find_faces(image, roi=roi, scale=scale)[1][0]
        threshold = 0.8
        self.assertGreaterEqual(get_iou(bb1), threshold)
        self.assertGreaterEqual(get_iou(bb2), threshold)
        self.assertGreaterEqual(get_iou(bb3), threshold)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_compute_embedding_vector_dimension(self):
        bubbles1 = imread(self.bubbles1)
        bb = self.test_engine.find_face(bubbles1)[1]
        embedding = self.test_engine.compute_embedding(bubbles1, bb)
        self.assertEqual(
            embedding.size, self.test_engine._embedder.embedding_dim)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_compute_embeddings_vectors_dimension(self):
        family = imread(self.family)
        bbs = self.test_engine.find_faces(family)[1]
        embeddings = self.test_engine.compute_embeddings(family, bbs)
        self.assertEqual(
            embeddings.shape[1], self.test_engine._embedder.embedding_dim)


if __name__ == '__main__':
    unittest.main()
