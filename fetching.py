import os
from pathlib import Path
from urllib.request import urlretrieve

import tqdm

RESOURCES = Path(__file__).parent / 'resources'


def fetch_images():
    """Fetch test images"""

    extract_dir = RESOURCES / 'images'
    # make sure the dir exists
    if not os.path.isdir(extract_dir):
        os.makedirs(os.path.abspath(extract_dir))

    # Load images from guesswh0/storage repository
    url_root = 'https://github.com/guesswh0/storage/raw/master/images/'
    for name in [
        "drive.jpg",
        "family.jpg",
        'bubbles1.jpg',
        'bubbles2.jpg'
    ]:
        # check if files exists
        file = os.path.join(extract_dir, name)
        if os.path.isfile(file):
            continue
        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                       desc=f"Downloading image: {name}") as t:
            reporthook = _tqdm_hook(t)
            filename, _ = urlretrieve(url_root + name, None, reporthook)
        os.replace(filename, file)


def fetch_models():
    """Fetch default dlib models"""

    extract_dir = RESOURCES / 'data'
    # make sure the dir exists
    if not os.path.isdir(extract_dir):
        os.makedirs(os.path.abspath(extract_dir))

    # Load dlib models
    url_root = "http://dlib.net/files/"
    for name in [
        "mmod_human_face_detector.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat.bz2",
        "shape_predictor_5_face_landmarks.dat.bz2"
    ]:
        # check if file exists
        file = os.path.join(extract_dir, name)
        if os.path.isfile(file[:-4]):
            continue

        with tqdm.tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                       desc=f"Downloading model: {name}") as t:
            reporthook = _tqdm_hook(t)
            filename, _ = urlretrieve(url_root + name, None, reporthook)
        os.replace(filename, file)
        unpack_archive(file, extract_dir)
        os.remove(file)


# from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
def _tqdm_hook(t):
    """Wraps tqdm instance"""

    last_b = [0]

    def update_to(blocknum=1, bs=1, size=None):
        """
        blocknum  : int, optional
            Number of blocks transferred so far [default: 1].
        bs  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if size not in (None, -1):
            t.total = size
        t.update((blocknum - last_b[0]) * bs)
        last_b[0] = blocknum

    return update_to


def unpack_archive(filename, extract_dir=None):
    """shutil.unpack_archive wrapper to unpack ['.dat.bz2'] archive.

        :param filename: name of the archive.
        :param extract_dir:name of the target directory, where the archive
        is unpacked. If not provided, the current working directory is used.
        """
    import shutil

    # hardcoded for .dat.bz2
    if filename.endswith('.dat.bz2'):
        shutil.register_unpack_format(
            'bzip2', ['dat.bz2'], _unpack_bz2, [], "bzip2'ed dat file")
        shutil.unpack_archive(filename, extract_dir, 'bzip2')
        shutil.unregister_unpack_format('bzip2')
    else:
        shutil.unpack_archive(filename, extract_dir)


def _unpack_bz2(filename, extract_dir):
    import bz2

    filename = Path(filename)
    if not isinstance(extract_dir, Path):
        extract_dir = Path(extract_dir)
    with open(filename, 'rb') as archive:
        data = bz2.decompress(archive.read())
        with open(extract_dir / filename.stem, 'wb') as file:
            file.write(data)
