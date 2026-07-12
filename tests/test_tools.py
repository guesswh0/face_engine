import io
import unittest

import numpy as np

from tests import TestCase

from face_engine.tools import imread


class TestImread(TestCase):

    def test_imread_file_path(self):
        image = imread(self.bubbles1)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.ndim, 3)

    def test_imread_bytes(self):
        with open(self.bubbles1, "rb") as file:
            data = file.read()
        np.testing.assert_array_equal(imread(data), imread(self.bubbles1))

    def test_imread_file_object(self):
        with open(self.bubbles1, "rb") as file:
            data = file.read()
        np.testing.assert_array_equal(imread(io.BytesIO(data)), imread(self.bubbles1))

    def test_imread_mode_conversion(self):
        image = imread(self.bubbles1, mode="L")
        self.assertEqual(image.ndim, 2)


if __name__ == "__main__":
    unittest.main()
