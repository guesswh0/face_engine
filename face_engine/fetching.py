"""
Fetching tools.
Used to download and unpack project models and testing data.
"""

import hashlib
import logging
import os
import tempfile
from urllib.request import Request, urlopen

import platformdirs
import tqdm

RESOURCES = platformdirs.user_cache_dir("face_engine")

_CHUNK_SIZE = 64 * 1024

# pinned sha256 checksums for known model files and test assets,
# keyed by origin filename; computed with `shasum -a 256 <file>`
# after a fresh download of each file
KNOWN_HASHES = {
    # dlib models (https://github.com/davisking/dlib-models)
    "mmod_human_face_detector.dat.bz2": "db9e9e40f092c118d5eb3e643935b216838170793559515541c56a2b50d9fc84",  # noqa: E501
    "dlib_face_recognition_resnet_model_v1.dat.bz2": "abb1f61041e434465855ce81c2bd546e830d28bcbed8d27ffbe5bb408b11553a",  # noqa: E501
    "shape_predictor_5_face_landmarks.dat.bz2": "6e787bbebf5c9efdb793f6cd1f023230c4413306605f24f299f12869f95aa472",  # noqa: E501
    # insightface model packs (v0.7 github release assets)
    "buffalo_l.zip": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",  # noqa: E501
    "antelopev2.zip": "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",  # noqa: E501
    # minifasnet anti-spoofing models (face-engine release assets,
    # exported from minivision-ai/Silent-Face-Anti-Spoofing checkpoints
    # by extra/export_minifasnet.py)
    "2.7_80x80_MiniFASNetV2.onnx": "3052e9d1d97270f5d9f197bed4f039cae23b8c14d6282e879ca3a63073792a97",  # noqa: E501
    "4_0_0_80x80_MiniFASNetV1SE.onnx": "2348be428f787149bf28dca49f802bd7ab9280ff3333475f76df1aafd2dada4f",  # noqa: E501
    # test assets (https://github.com/guesswh0/storage)
    "test.zip": "7e967899f1a106908798d750f81d3a966422300b479757b600eaf3ea4d96723b",
    "train.zip": "d61d21d6b0198df9b2f045de784924984177042a595a1a5fd42c3771b2b24b15",
    "bubbles1.jpg": "5b46c2103d9dc305f73fc8c941cc3e1947897c5d94020290199faeb4bf9330f9",
    "bubbles2.jpg": "cd0b7bcce499397b2cbeaf29ca194b176134624b46f5353e15447c84f66402ab",
    "family.jpg": "4082058882f9b96236530be5e176586557d565f18ded7175021ca49cf56bbf0f",
    "drive.jpg": "cc09ecced9d570437ad9b60e6b3d8387ba84cba99a7b24f74263661344fc531c",
    "book_stack.jpg": "9d3138ae53c8d956a160f389929114ef9ffbefdbeb2ae23421d9d38cf711c517",
}


def fetch_file(url, extract_dir=None, sha256=None):
    """Fetch file by URL to extract_dir folder.

    The downloaded file's SHA-256 checksum is verified against the
    ``sha256`` argument, falling back to the :data:`KNOWN_HASHES` entry
    for the file name. If neither is available verification is skipped.

    :param url: http(s) URL of the file to download
    :param extract_dir: target directory, defaults to :data:`RESOURCES`
    :param sha256: expected hex-encoded SHA-256 checksum of the file
    :raises ValueError: on non-http(s) URL scheme
    :raises IOError: on checksum mismatch
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"unsupported URL scheme: {url}")
    if not url.startswith("https://"):
        logging.warning(
            f"Fetching file from insecure URL: {url}. "
            "Consider using HTTPS to prevent Man-in-the-Middle attacks."
        )

    if not extract_dir:
        extract_dir = RESOURCES

    # make sure the dir exists
    extract_dir = os.path.abspath(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    origin = url.split("/")[-1]
    # check if file exists (verified at download time, not re-hashed)
    file = os.path.join(extract_dir, origin)
    if os.path.exists(file):
        return

    if sha256 is None:
        sha256 = KNOWN_HASHES.get(origin)
        if sha256 is None:
            logging.debug("no known sha256 for %s, skipping verification", origin)

    # download to a temp file in the same directory so os.replace is atomic
    request = Request(url, headers={"User-Agent": "face-engine/3.0"})
    digest = hashlib.sha256()
    temp = tempfile.NamedTemporaryFile(dir=extract_dir, delete=False)
    try:
        with urlopen(request, timeout=30) as response, temp:
            size = response.headers.get("Content-Length")
            with tqdm.tqdm(
                total=int(size) if size else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=f"Downloading file: {origin}",
            ) as t:
                while chunk := response.read(_CHUNK_SIZE):
                    temp.write(chunk)
                    digest.update(chunk)
                    t.update(len(chunk))
        if sha256 and digest.hexdigest() != sha256:
            raise IOError(
                f"sha256 mismatch for {origin}: "
                f"expected {sha256}, got {digest.hexdigest()}. "
                "The upstream file may have changed, please report an issue "
                "at https://github.com/guesswh0/face_engine/issues."
            )
    except BaseException:
        os.remove(temp.name)
        raise
    os.replace(temp.name, file)
    for ext in [".bz2", ".zip", ".tar", ".gz"]:
        if origin.endswith(ext):
            unpack_archive(file, extract_dir)


def unpack_archive(filename, extract_dir=None):
    """shutil.unpack_archive wrapper to unpack ['.dat.bz2'] archive.

    :param filename: name of the archive.
    :param extract_dir: name of the target directory, where the archive
        is unpacked. If not provided, the current working directory is used.
    """
    import shutil

    # hardcoded for .dat.bz2
    if filename.endswith(".dat.bz2"):
        shutil.register_unpack_format(
            "bzip2", ["dat.bz2"], _unpack_bz2, [], "bzip2'ed dat file"
        )
        shutil.unpack_archive(filename, extract_dir, "bzip2")
        shutil.unregister_unpack_format("bzip2")
    else:
        shutil.unpack_archive(filename, extract_dir)


def _unpack_bz2(filename, extract_dir):
    import bz2

    with open(filename, "rb") as archive:
        data = bz2.decompress(archive.read())
        with open(
            os.path.join(extract_dir, os.path.basename(filename)[:-4]), "wb"
        ) as file:
            file.write(data)
