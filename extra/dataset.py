#!/usr/bin/env python3
# $ dataset.py [-c 10] [-i 10] [--split 0.7] [--shuffle True] source target

import argparse
import csv
import os
import pickle
import shutil

from extra.tools import get_face_dataset
from face_engine import FaceEngine


def load_dataset(dir):
    images = []
    names = []
    bounding_boxes = []
    embeddings = []
    for name in sorted(os.listdir(dir)):
        file = os.path.join(dir, name)
        if os.path.isdir(file):
            paths = [os.path.join(file, i) for i in sorted(os.listdir(file))]
            images += paths
            names += [name] * len(paths)
        else:
            if name.endswith('.csv'):
                with open(file, newline='') as csv_file:
                    reader = csv.reader(csv_file)
                    # next(reader, None)
                    bounding_boxes += [tuple(map(int, row)) for row in reader]
            if name.endswith('.dat'):
                with open(file, 'rb') as dat_file:
                    embeddings += pickle.load(dat_file)
    return images, names, bounding_boxes, embeddings


def create_dataset(dirname, image_paths, bounding_boxes, embeddings):
    # copy images
    for image_path in image_paths:
        shutil.copy(image_path, dirname)
    # create csv file with bounding boxes
    with open(dirname + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(('left', 'upper', 'right', 'lower'))
        writer.writerows(bounding_boxes)
    # create .dat file with pre-calculated embeddings
    with open(dirname + '.dat', 'wb') as dat:
        pickle.dump(embeddings, dat)


def prepare_dataset(source, target, classes=10, images=10, split=0.7, shuffle=False):
    assert split <= 1.0, "split must be less than or equal to 1.0"
    engine = FaceEngine()
    target = os.path.expanduser(target)
    train_dir = os.path.join(target, 'train')
    test_dir = os.path.join(target, 'test')
    # create dirs if not exist
    if not os.path.exists(target):
        os.makedirs(target)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    classes_total = 0
    for face_images in get_face_dataset(source, engine, shuffle):
        if classes_total >= classes:
            break
        if face_images.size < images:
            continue
        train_name = os.path.join(train_dir, face_images.name)
        test_name = os.path.join(test_dir, face_images.name)
        if not os.path.exists(train_name):
            os.makedirs(train_name)
        if not os.path.exists(test_name):
            os.makedirs(test_name)
        # get required slices
        image_paths = face_images.image_paths[:images]
        bounding_boxes = face_images.bounding_boxes[:images]
        embeddings = face_images.embeddings[:images]
        # split and create datasets
        split = int(images * split)
        create_dataset(train_name, image_paths[:split], bounding_boxes[:split], embeddings[:split])
        create_dataset(test_name, image_paths[split:], bounding_boxes[split:], embeddings[split:])
        classes_total += 1
        print(face_images)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Source directory')
    parser.add_argument('target', type=str, help='Target directory')
    parser.add_argument('-c', '--classes', type=int, help='Number of classes', default=10)
    parser.add_argument('-i', '--images', type=int, help='Images per class', default=10)
    parser.add_argument('--split', type=float, help='split_ratio', default=0.7)
    parser.add_argument('--shuffle', type=bool, help='use random shuffle', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    prepare_dataset(args.source, args.target, args.classes, args.images, args.split, args.shuffle)
