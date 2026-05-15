import unittest
import os
import shutil
import tempfile
import numpy as np
from face_engine.models.basic_estimator import BasicEstimator

class TestBasicEstimatorPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.embeddings = np.random.rand(10, 128).astype(np.float32)
        self.class_names = [f"person_{i}" for i in range(10)]
        self.estimator = BasicEstimator()
        self.estimator.fit(self.embeddings, self.class_names)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_load(self):
        self.estimator.save(self.test_dir)

        new_estimator = BasicEstimator()
        new_estimator.load(self.test_dir)

        np.testing.assert_array_equal(new_estimator.embeddings, self.embeddings)
        self.assertEqual(new_estimator.class_names, self.class_names)
        np.testing.assert_array_equal(new_estimator.fitted_norms_sq, self.estimator.fitted_norms_sq)

        # Test prediction after load
        test_emb = np.random.rand(1, 128).astype(np.float32)
        scores1, classes1 = self.estimator.predict(test_emb)
        scores2, classes2 = new_estimator.predict(test_emb)

        self.assertEqual(classes1, classes2)
        np.testing.assert_allclose(scores1, scores2)

    def test_load_legacy(self):
        # Manually create a pickle state without fitted_norms_sq
        state = {
            "embeddings": self.embeddings,
            "class_names": self.class_names,
            "name": "basic"
        }
        import pickle
        with open(os.path.join(self.test_dir, "basic.estimator.p"), "wb") as f:
            pickle.dump(state, f)

        new_estimator = BasicEstimator()
        new_estimator.load(self.test_dir)

        self.assertTrue(hasattr(new_estimator, "fitted_norms_sq"))
        np.testing.assert_array_equal(new_estimator.fitted_norms_sq, np.sum(np.square(self.embeddings), axis=1))

if __name__ == "__main__":
    unittest.main()
