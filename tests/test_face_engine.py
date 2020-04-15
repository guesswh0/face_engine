import os
import unittest

import numpy as np
from PIL import Image

try:
    import dlib
except ImportError:
    dlib = None
else:
    from face_engine import fetching

    # fetch if dlib installed
    fetching.fetch_models()
    fetching.fetch_images()

from face_engine import FaceEngine, RESOURCES
from face_engine.exceptions import FaceError, TrainError


class TestFaceEngine(unittest.TestCase):

    def setUp(self) -> None:
        images = os.path.join(RESOURCES, 'images')
        self.images = [
            os.path.join(images, image) for image in sorted(os.listdir(images))
        ]
        self.bubbles1 = os.path.join(images, 'bubbles1.jpg')
        self.bubbles2 = os.path.join(images, 'bubbles2.jpg')
        self.cat = os.path.join(images, 'cat.jpg')
        self.dog = os.path.join(images, 'dog.jpg')
        self.drive = os.path.join(images, 'drive.jpg')
        self.family = os.path.join(images, 'family.jpg')
        self.default_engine = FaceEngine()

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_init_default_models(self):
        with self.subTest(detector='hog'):
            self.assertEqual(self.default_engine.detector, 'hog')
        with self.subTest(embedder='resnet'):
            self.assertEqual(self.default_engine.embedder, 'resnet')
        with self.subTest(predictor='linear'):
            self.assertEqual(self.default_engine.predictor, 'linear')

    @unittest.skipIf(dlib, "dlib package is installed")
    def test_init_implicit_abstract_models(self):
        with self.subTest(detector='abstract_detector'):
            self.assertEqual(self.default_engine.detector, 'abstract_detector')
        with self.subTest(embedder='abstract_embedder'):
            self.assertEqual(self.default_engine.embedder, 'abstract_embedder')

    def test_init_explicit_abstract_models(self):
        engine = FaceEngine(
            detector='abstract_detector',
            embedder='abstract_embedder',
            predictor='abstract_predictor'
        )
        with self.subTest(detector='abstract_detector'):
            self.assertEqual(engine.detector, 'abstract_detector')
        with self.subTest(embedder='abstract_embedder'):
            self.assertEqual(engine.embedder, 'abstract_embedder')
        with self.subTest(predictor='abstract_predictor'):
            self.assertEqual(engine.predictor, 'abstract_predictor')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_init_limit_param(self):
        engine = FaceEngine(limit=2)
        with self.assertRaises(TrainError):
            images = [self.bubbles1, self.bubbles2, self.family]
            classes = [0, 0, 1]
            engine.fit(images, classes)

    def test_setter_detector_model(self):
        engine = FaceEngine()
        engine.detector = 'abstract_detector'
        self.assertEqual(engine.detector, 'abstract_detector')

    def test_setter_embedder_model(self):
        engine = FaceEngine()
        engine.embedder = 'abstract_embedder'
        self.assertEqual(engine.embedder, 'abstract_embedder')

    def test_setter_predictor_model(self):
        engine = FaceEngine()
        engine.predictor = 'abstract_predictor'
        self.assertEqual(engine.predictor, 'abstract_predictor')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_setter_dlib_detector_model(self):
        engine = FaceEngine()
        engine.detector = 'mmod'
        self.assertEqual(engine.detector, 'mmod')

    def test_getter_detector_model(self):
        engine = FaceEngine()
        engine.detector = 'abstract_detector'
        self.assertIsInstance(engine.detector, str)

    def test_getter_embedder_model(self):
        engine = FaceEngine()
        engine.embedder = 'abstract_embedder'
        self.assertIsInstance(engine.embedder, str)

    def test_getter_predictor_model(self):
        engine = FaceEngine()
        engine.predictor = 'abstract_predictor'
        self.assertIsInstance(engine.predictor, str)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_fit(self):
        images = [self.bubbles1, self.bubbles2, self.family]
        classes = [0, 0, 1]
        self.default_engine.fit(images, classes)
        with self.subTest(n_samples=3):
            self.assertEqual(self.default_engine.n_samples, 3)
        with self.subTest(n_identities=2):
            self.assertEqual(self.default_engine.n_identities, 2)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_fit_with_bounding_boxes(self):
        images = [self.bubbles1, self.bubbles2, self.family]
        classes = [0, 0, 1]
        bbs = []
        for img in images:
            _, bb = self.default_engine.find_face(np.asarray(Image.open(img)))
            bbs.append(bb)
        self.default_engine.fit(images, classes, bbs)
        with self.subTest(n_samples=3):
            self.assertEqual(self.default_engine.n_samples, 3)
        with self.subTest(n_identities=2):
            self.assertEqual(self.default_engine.n_identities, 2)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_returns_bounding_box(self):
        bubbles1 = np.asarray(Image.open(self.bubbles1))
        _, bb = self.default_engine.find_face(bubbles1)
        # returns single bounding box
        self.assertEqual(len(bb), 4)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_normalize_param(self):
        family = np.asarray(Image.open(self.family))
        _, bb = self.default_engine.find_face(family, normalize=True)
        for point in bb:
            with self.subTest():
                self.assertLess(point, 1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_face_raises_face_error(self):
        cat = np.asarray(Image.open(self.cat))
        with self.assertRaises(FaceError):
            _, bb = self.default_engine.find_face(cat)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_returns_bounding_boxes(self):
        # family image has three faces
        family = np.asarray(Image.open(self.family))
        _, bbs = self.default_engine.find_faces(family)
        self.assertGreaterEqual(len(bbs), 1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_roi_param(self):
        bubbles1 = np.asarray(Image.open(self.bubbles1))
        h, w = bubbles1.shape[:2]
        with self.subTest(roi='first_half'):
            _, bbs = self.default_engine.find_faces(
                bubbles1, roi=(0, 0, w, h // 2))
            self.assertEqual(len(bbs), 1)
        with self.subTest(roi='second_half'):
            with self.assertRaises(FaceError):
                _, bbs = self.default_engine.find_faces(
                    bubbles1, roi=(0, h // 2, w, h))

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_normalize_param(self):
        family = np.asarray(Image.open(self.family))
        _, bbs = self.default_engine.find_faces(family, normalize=True)
        # check only first bounding box
        for point in bbs[0]:
            with self.subTest():
                self.assertLess(point, 1)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_find_faces_raises_face_error(self):
        cat = np.asarray(Image.open(self.cat))
        with self.assertRaises(FaceError):
            _, bbs = self.default_engine.find_faces(cat)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_compute_embedding_vector_dimension(self):
        bubbles1 = np.asarray(Image.open(self.bubbles1))
        _, bb = self.default_engine.find_face(bubbles1)
        embedding = self.default_engine.compute_embedding(bubbles1, bb)
        self.assertEqual(
            embedding.size, self.default_engine._embedder.embedding_dim
        )

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_compute_embeddings_vectors_dimension(self):
        family = np.asarray(Image.open(self.family))
        _, bbs = self.default_engine.find_faces(family)
        embeddings = self.default_engine.compute_embeddings(family, bbs)
        self.assertEqual(
            embeddings.shape[1], self.default_engine._embedder.embedding_dim
        )


if __name__ == '__main__':
    unittest.main()
