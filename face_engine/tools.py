# Copyright 2020 Daniyar Kussainov
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

import re
from importlib import util
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image

from . import logger

# from django URLValidator
_url_validation_regex = re.compile(
    r'^(?:http|ftp)s?://'  # scheme
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$',
    re.IGNORECASE)


def imread(uri, mode=None):
    """ Reads an image from the specified file.

    References:
        [1] https://pillow.readthedocs.io/en/stable/reference/Image.html
        [2] https://stackoverflow.com/questions/7160737


    :param uri: image file, file path or url
    :type uri:  str | bytes | file | os.PathLike

    :param mode: format mode to convert the image to.
        More info on accepted mode types see on [2].
    :type mode: str

    :returns: image content as unsigned 8-bit numpy array
    """

    # handle requests too
    if isinstance(uri, Request):
        uri = uri.full_url

    # url validation
    if isinstance(uri, str) and re.match(_url_validation_regex, uri):
        uri = urlopen(uri)

    image = Image.open(uri)
    if mode:
        image = image.convert(mode)
    return np.uint8(image)


def import_module(filepath):
    """Convenient function to import module by given filepath.

    :param filepath: absolute or relative filepath
    :type filepath: str | bytes | os.PathLike | Path
    """

    path = Path(filepath)
    try:
        spec = util.spec_from_file_location(path.stem, filepath)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        logger.info("Module '%s' has not been imported: %s", path.stem, e)


def import_submodules(filepath):
    """Convenient function to import all submodules of given filepath.

    :param filepath: absolute or relative filepath
    :type filepath: str | bytes | os.PathLike | Path
    """

    base = Path(filepath).parent
    for file in base.glob('*.py'):
        if not file.stem.startswith('_'):
            import_module(file)
