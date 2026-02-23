"""
Fetching tools.
Used to download and unpack project models and testing data.
"""

import os
import shutil
import warnings
from typing import Optional

import platformdirs
import tqdm

RESOURCES = platformdirs.user_cache_dir("face_engine")


def fetch_file(url: str, extract_dir: Optional[str] = None) -> None:
    """Fetch file by URL to extract_dir folder"""
    if not extract_dir:
        extract_dir = RESOURCES

    # make sure the dir exists
    os.makedirs(os.path.abspath(extract_dir), exist_ok=True)

    origin = url.split("/")[-1]
    # check if file exists
    file_path = os.path.join(extract_dir, origin)
    if os.path.exists(file_path):
        return

    # Try using requests first (more robust)
    try:
        import requests

        _fetch_with_requests(url, file_path, origin)
    except ImportError:
        # Fallback to urllib
        _fetch_with_urllib(url, file_path, origin)

    # Unpack if needed
    for ext in [".bz2", ".zip", ".tar", ".gz"]:
        if origin.endswith(ext):
            unpack_archive(file_path, extract_dir)


def _fetch_with_requests(url: str, file_path: str, desc: str) -> None:
    import requests

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with tqdm.tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {desc}"
    ) as progress_bar:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))


def _fetch_with_urllib(url: str, file_path: str, desc: str) -> None:
    from urllib.request import urlretrieve, build_opener, install_opener

    # Add User-Agent to avoid 403 Forbidden on some sites
    opener = build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    install_opener(opener)

    with tqdm.tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=f"Downloading {desc}",
    ) as t:
        reporthook = _tqdm_hook(t)
        temp, _ = urlretrieve(url, None, reporthook)
        shutil.move(temp, file_path)


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


def unpack_archive(filename: str, extract_dir: Optional[str] = None) -> None:
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
