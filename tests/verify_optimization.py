import sys
from unittest.mock import MagicMock, patch
import unittest

# Mock numpy
mock_np = MagicMock()
sys.modules['numpy'] = mock_np

# Mock other dependencies
sys.modules['PIL'] = MagicMock()
sys.modules['platformdirs'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

# Now import the class under test
from face_engine.models.basic_estimator import BasicEstimator

class TestBasicEstimatorVectorization(unittest.TestCase):
    def test_fit_calls_asarray(self):
        estimator = BasicEstimator()
        data = [[1, 2], [3, 4]]
        estimator.fit(data, ["p1", "p2"])
        mock_np.asarray.assert_called_with(data)

    def test_predict_vectorized_calls(self):
        estimator = BasicEstimator()
        # Setup fitted data
        fitted_data = MagicMock()
        fitted_data.__pow__.return_value = fitted_data
        estimator.embeddings = fitted_data
        estimator.class_names = ["p1", "p2"]

        # Input data
        input_data = MagicMock()
        input_data.__len__.return_value = 1
        input_data.__pow__.return_value = input_data

        # Reset mock to clear calls from setup
        mock_np.reset_mock()
        mock_np.asarray.return_value = input_data

        scores, names = estimator.predict(input_data)

        # Check that expected vectorized operations were called
        # np.asarray
        mock_np.asarray.assert_called()
        # np.sum(embeddings**2, ...)
        mock_np.sum.assert_any_call(input_data, axis=1, keepdims=True)
        # np.sum(self.embeddings**2, ...)
        mock_np.sum.assert_any_call(fitted_data, axis=1)
        # np.dot(embeddings, self.embeddings.T)
        mock_np.dot.assert_called()
        # np.maximum
        mock_np.maximum.assert_called()
        # np.argmin
        mock_np.argmin.assert_called()
        # np.exp
        mock_np.exp.assert_called()

    def test_predict_empty_input(self):
        estimator = BasicEstimator()
        estimator.class_names = ["p1"]
        scores, names = estimator.predict([])
        self.assertEqual(scores, [])
        self.assertEqual(names, [])

if __name__ == "__main__":
    unittest.main()
