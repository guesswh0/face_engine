# Copyright 2019 Daniyar Kussainov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import PurePosixPath

from imageio.core.fetching import _fetch_file


# Adapted from imageio:
# https://github.com/imageio/imageio/blob/master/imageio/core/util.py
def get_remote_file(filename, extract_dir, url_root=None):
    """ Get a the path for the local version of a file from the web

    :param filename: The relative file name on the remote data repository
        to download.
    :type filename: str

    :param extract_dir: The directory where the file will be cached
        if a download was required to obtain the file. If the directory
        does not exist, it will be created.
    :type extract_dir: str
    :param url_root: Default [None] correspond to paths on
        ``https://github.com/guesswh0/storage``.
    :type url_root: str

    :returns: The path to the file on the local system.
    :rtype: str
    """

    if not url_root:
        # Default url
        url_root = "https://github.com/guesswh0/storage/raw/master/"
    url = url_root + filename
    native_filename = os.path.normcase(filename)  # convert to native
    file = os.path.join(extract_dir, native_filename)
    # check if files exists
    if os.path.isfile(file):
        return
    if file[-4:] == '.bz2' and os.path.isfile(file[:-4]):
        return
    # make sure the dir exists
    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.abspath(os.path.dirname(file)))
    _fetch_file(url, file)
    return file


def unpack_archive(file_name, extract_dir):
    """Convenient wrapper to unpack [.dat.bz2] archive."""

    pps = PurePosixPath(file_name)
    if '.bz2' in pps.suffixes:
        # Hardcoded to dat.bz2 archive extension only
        if '.dat' in pps.suffixes:
            shutil.unregister_unpack_format('bztar')
            shutil.register_unpack_format(
                'bztar', ['dat.bz2'], _unpack_bz2, [], "bzip2'ed tar-file")
    shutil.unpack_archive(
        os.path.join(extract_dir, file_name),
        os.path.join(extract_dir, pps.parent.name))


def _unpack_bz2(file_name, extract_dir):
    import bz2

    pps = PurePosixPath(file_name)
    file = os.path.join(extract_dir, pps.stem)

    with open(file_name, 'rb') as archive:
        data = bz2.decompress(archive.read())
        with open(file, 'wb') as f:
            f.write(data)
