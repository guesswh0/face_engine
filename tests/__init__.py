import logging
import os
import unittest

from face_engine import RESOURCES

try:
    import dlib
except ImportError:
    dlib = None
else:
    from face_engine.fetching import fetch_file

    # fetch datasets
    base = "https://github.com/guesswh0/storage/raw/master/"
    extract_dir = os.path.join(RESOURCES, 'datasets')
    fetch_file(base + "datasets/test.zip", extract_dir)
    fetch_file(base + "datasets/train.zip", extract_dir)

    # fetch images
    extract_dir = os.path.join(RESOURCES, 'images')
    fetch_file(base + 'images/bubbles1.jpg', extract_dir)
    fetch_file(base + 'images/bubbles2.jpg', extract_dir)
    fetch_file(base + 'images/family.jpg', extract_dir)
    fetch_file(base + 'images/drive.jpg', extract_dir)
    fetch_file(base + 'images/book_stack.jpg', extract_dir)


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
