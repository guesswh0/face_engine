import logging
import os
import unittest

try:
    import dlib
except ImportError:
    dlib = None
else:
    from face_engine import fetching

    # fetch if dlib installed
    fetching.fetch_models()
    fetching.fetch_images()
    fetching.fetch_datasets()

from face_engine import RESOURCES


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        images = os.path.join(RESOURCES, 'images')
        cls.bubbles1 = os.path.join(images, 'bubbles1.jpg')
        cls.bubbles2 = os.path.join(images, 'bubbles2.jpg')
        cls.book_stack = os.path.join(images, 'book_stack.jpg')
        cls.drive = os.path.join(images, 'drive.jpg')
        cls.family = os.path.join(images, 'family.jpg')
        cls.train = os.path.join(RESOURCES, 'datasets', 'train')
        cls.test = os.path.join(RESOURCES, 'datasets', 'test')

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)
