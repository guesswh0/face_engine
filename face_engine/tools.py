import re
from importlib import util
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image

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
    """Reads an image from the specified file.

    References:
        1. https://pillow.readthedocs.io/en/stable/reference/Image.html
        2. https://stackoverflow.com/questions/7160737


    :param uri: image file, file path or url
    :type uri: Union[str, bytes, file, os.PathLike]

    :param mode: format mode to convert the image to.
        More info on accepted mode types see on [1].
    :type mode: str

    :returns: image content as unsigned 8-bit numpy array
    :rtype: :class:`~numpy.uint8`
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

    Note that this function is using file path unlike importlib's import_module.

    :param filepath: absolute or relative filepath
    :type filepath: Union[str, bytes, os.PathLike, Path]
    """

    path = Path(filepath)
    try:
        spec = util.spec_from_file_location(path.stem, filepath)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError:
        # skip raising on project [optional] default models
        if path.stem not in ['dlib_models', 'insightface_models']:
            raise


def import_package(package):
    """Convenient function to import all modules of given package
    except __init__.py

    :param package: absolute or relative filepath to the package
    :type package: Union[str, bytes, os.PathLike, Path]
    """

    for file in Path(package).parent.glob('*.py'):
        # skip all underscore files (__init__.py, compiled files, etc)
        if not file.stem.startswith('_'):
            import_module(file)
