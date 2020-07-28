import os

import numpy as np


class ImageClass:
    """Stores the paths to images for a given class"""

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
        self.size = len(self.image_paths)

    def __str__(self):
        return self.name + ', ' + str(self.size) + ' images'

    def __len__(self):
        return self.size


class FaceImage(ImageClass):
    """Extend ImageClass with face specific attributes"""

    def __init__(self, name, image_paths, bounding_boxes, embeddings):
        super().__init__(name, image_paths)
        self.bounding_boxes = bounding_boxes
        self.embeddings = embeddings


def is_image(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return '.' in filename and ext in ['jpg', 'jpeg', 'png']


def get_image_paths(dir, shuffle=False):
    image_paths = []
    if os.path.isdir(dir):
        images = [name for name in os.listdir(dir) if is_image(name)]
        if shuffle:
            np.random.shuffle(images)
        else:
            images.sort()
        image_paths = [os.path.join(dir, image) for image in images]
    return image_paths


def get_image_dataset(dir, shuffle=False):
    dataset = []
    dir = os.path.expanduser(dir)
    class_names = [name for name in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, name))]
    if shuffle:
        np.random.shuffle(class_names)
    else:
        class_names.sort()
    for class_name in class_names:
        image_dir = os.path.join(dir, class_name)
        image_paths = get_image_paths(image_dir, shuffle)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset


def get_face_dataset(dir, engine, shuffle=False):
    """Generator of FaceImages"""
    from face_engine.exceptions import FaceNotFoundError
    from face_engine.tools import imread
    dataset = get_image_dataset(dir, shuffle)
    if shuffle:
        np.random.shuffle(dataset)
    for image in dataset:
        bounding_boxes = []
        embeddings = []
        image_paths = image.image_paths
        for i, image_path in enumerate(image_paths):
            img = imread(image_path)
            try:
                bounding_box = engine.find_face(img)[1]
                bounding_boxes.append(bounding_box)
                embeddings.append(engine.compute_embedding(img, bounding_box))
            except FaceNotFoundError:
                del image_paths[i]
                continue
        yield FaceImage(image.name, image_paths, bounding_boxes, embeddings)


def filter_dataset(dataset, min_images=5):
    filtered_dataset = []
    for cls in dataset:
        if cls.size >= min_images:
            filtered_dataset.append(cls)
    return filtered_dataset
