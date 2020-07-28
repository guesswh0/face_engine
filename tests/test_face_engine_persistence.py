import os
import unittest

from tests import TestCase, dlib

from face_engine import FaceEngine, load_engine


class TestFaceEnginePersistence(TestCase):

    def setUp(self):
        self.test_engine = FaceEngine()

    def tearDown(self):
        if os.path.isfile('test_engine.p'):
            os.remove('test_engine.p')
        if os.path.isfile('basic.estimator.p'):
            os.remove('basic.estimator.p')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_save(self):
        self.test_engine.save('test_engine.p')
        self.assertEqual(os.path.isfile('test_engine.p'), True)

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_save_with_fitted_engine(self):
        images = [self.bubbles1, self.bubbles2]
        classes = [0, 0]
        self.test_engine.fit(images, classes)
        self.test_engine.save('test_engine.p')
        self.assertEqual(os.path.isfile('test_engine.p'), True)
        self.assertEqual(os.path.isfile('basic.estimator.p'), True)

    @unittest.skipIf(dlib, "dlib package is installed")
    def test_load_engine_without_dlib(self):
        self.test_engine.save('test_engine.p')
        engine = load_engine('test_engine.p')
        self.assertIsInstance(engine, FaceEngine)
        self.assertEqual(engine.detector, 'abstract_detector')
        self.assertEqual(engine.embedder, 'abstract_embedder')
        self.assertEqual(engine.estimator, 'basic')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_load_engine_with_dlib(self):
        self.test_engine.save('test_engine.p')
        engine = load_engine('test_engine.p')
        self.assertIsInstance(engine, FaceEngine)
        self.assertEqual(engine.detector, 'hog')
        self.assertEqual(engine.embedder, 'resnet')
        self.assertEqual(engine.estimator, 'basic')

    @unittest.skipUnless(dlib, "dlib package is not installed")
    def test_load_engine_with_estimator_state(self):
        images = [self.bubbles1, self.bubbles2]
        classes = [0, 0]
        self.test_engine.fit(images, classes)
        self.test_engine.save('test_engine.p')
        engine = load_engine('test_engine.p')
        self.assertIsInstance(engine, FaceEngine)
        self.assertEqual(engine.detector, 'hog')
        self.assertEqual(engine.embedder, 'resnet')
        self.assertEqual(engine.estimator, 'basic')
        self.assertEqual(engine.n_classes, 1)
        self.assertEqual(engine.n_samples, 2)


if __name__ == '__main__':
    unittest.main()
