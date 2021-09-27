"""
Fetching tools.
Used to download and unpack project models and testing data.
"""

import os
from urllib.request import urlretrieve

import tqdm

RESOURCES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'resources'))


def fetch_file(url, extract_dir=None):
    """Fetch file by URL to extract_dir folder"""
    if not extract_dir:
        extract_dir = RESOURCES
    else:
        # make sure the dir exists
        if not os.path.isdir(extract_dir):
            os.makedirs(os.path.abspath(extract_dir))
    origin = url.split('/')[-1]
    # check if file exists
    file = os.path.join(extract_dir, origin)
    if os.path.exists(file):
        return
    # download file
    with tqdm.tqdm(
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"Downloading file: {origin}"
    ) as t:
        reporthook = _tqdm_hook(t)
        temp, _ = urlretrieve(url, None, reporthook)
    os.replace(temp, file)
    for ext in ['.bz2', '.zip', '.tar', 'gz']:
        if origin.endswith(ext):
            unpack_archive(file, extract_dir)


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
    :param extract_dir: name of the target directory, where the archive
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

    with open(filename, 'rb') as archive:
        data = bz2.decompress(archive.read())
        with open(os.path.join(extract_dir, os.path.basename(filename)[:-4]),
                  'wb') as file:
            file.write(data)
