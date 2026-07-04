import json
import os
import pickle
import unittest

import numpy as np

from tests import TestCase, dlib, insightface

from face_engine import FaceEngine, load_engine
from face_engine.models.basic_estimator import BasicEstimator


class TestFaceEnginePersistence(TestCase):

    filename = "test_engine.json"
    artifacts = (
        "test_engine.json",
        "test_engine.p",
        "basic.estimator.npz",
        "basic.estimator.json",
        "basic.estimator.p",
    )

    def setUp(self):
        self.test_engine = FaceEngine()

    def tearDown(self):
        for name in self.artifacts:
            if os.path.isfile(name):
                os.remove(name)

    def test_save(self):
        self.test_engine.save(self.filename)
        self.assertTrue(os.path.isfile(self.filename))
        with open(self.filename) as file:
            data = json.load(file)
        self.assertEqual(data["format"], "face-engine")
        self.assertEqual(data["detector"], self.test_engine.detector)

    @unittest.skipUnless(dlib or insightface, "no face recognition backend installed")
    def test_save_with_fitted_engine(self):
        self.test_engine.fit([self.bubbles1, self.bubbles2], [0, 0])
        self.test_engine.save(self.filename)
        self.assertTrue(os.path.isfile(self.filename))
        self.assertTrue(os.path.isfile("basic.estimator.npz"))
        self.assertTrue(os.path.isfile("basic.estimator.json"))

    def test_load_engine_restores_model_names(self):
        self.test_engine.save(self.filename)
        engine = load_engine(self.filename)
        self.assertIsInstance(engine, FaceEngine)
        self.assertEqual(engine.detector, self.test_engine.detector)
        self.assertEqual(engine.embedder, self.test_engine.embedder)
        self.assertEqual(engine.estimator, self.test_engine.estimator)
        self.assertEqual(engine.antispoof, self.test_engine.antispoof)

    def test_load_engine_pre31_file_without_antispoof(self):
        self.test_engine.save(self.filename)
        with open(self.filename) as file:
            data = json.load(file)
        # files saved by face-engine 3.0 have no antispoof key
        del data["antispoof"]
        with open(self.filename, "w") as file:
            json.dump(data, file)
        engine = load_engine(self.filename)
        self.assertEqual(engine.antispoof, "abstract_antispoof")

    @unittest.skipUnless(dlib or insightface, "no face recognition backend installed")
    def test_load_engine_with_estimator_state(self):
        self.test_engine.fit([self.bubbles1, self.bubbles2], [0, 0])
        self.test_engine.save(self.filename)
        engine = load_engine(self.filename)
        self.assertIsInstance(engine, FaceEngine)
        self.assertEqual(engine.n_classes, 1)
        self.assertEqual(engine.n_samples, 2)

    @unittest.skipUnless(dlib or insightface, "no face recognition backend installed")
    def test_round_trip_predictions_identical(self):
        self.test_engine.fit([self.bubbles1, self.bubbles2], [1, 2])
        bbs, class_names = self.test_engine.make_prediction(self.bubbles2)
        self.test_engine.save(self.filename)
        engine = load_engine(self.filename)
        restored_bbs, restored_class_names = engine.make_prediction(self.bubbles2)
        np.testing.assert_array_equal(bbs, restored_bbs)
        self.assertEqual(class_names, restored_class_names)

    def test_load_engine_legacy_pickle_raises(self):
        with open("test_engine.p", "wb") as file:
            pickle.dump({"detector": "hog"}, file)
        with self.assertRaises(RuntimeError) as context:
            load_engine("test_engine.p")
        self.assertIn("re-fit", str(context.exception))

    def test_load_engine_invalid_json_raises(self):
        with open(self.filename, "w") as file:
            file.write("not a json document")
        with self.assertRaises(RuntimeError):
            load_engine(self.filename)

    def test_load_engine_foreign_json_raises(self):
        with open(self.filename, "w") as file:
            json.dump({"detector": "hog"}, file)
        with self.assertRaises(TypeError):
            load_engine(self.filename)

    def test_estimator_legacy_pickle_raises(self):
        with open("basic.estimator.p", "wb") as file:
            pickle.dump({}, file)
        estimator = BasicEstimator()
        with self.assertRaises(RuntimeError) as context:
            estimator.load("")
        self.assertIn("Re-fit", str(context.exception))

    def test_estimator_int_class_names_survive(self):
        estimator = BasicEstimator()
        estimator.fit(np.random.rand(2, 128), [1, 2])
        estimator.save("")
        restored = BasicEstimator()
        restored.load("")
        self.assertEqual(restored.class_names, [1, 2])
        np.testing.assert_array_equal(restored.embeddings, estimator.embeddings)


if __name__ == "__main__":
    unittest.main()
