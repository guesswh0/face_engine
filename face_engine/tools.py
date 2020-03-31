import logging
import os
from importlib import util
from pathlib import Path

logger = logging.getLogger(__name__)


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
        logger.warning("Module '%s' has not been imported: %s", path.stem, e)


def import_submodules(filepath):
    """Convenient function to import all submodules of given filepath.

    :param filepath: absolute or relative filepath
    :type filepath: str | bytes | os.PathLike | Path
    """

    base = Path(filepath).parent
    for file in base.glob('*.py'):
        if not file.stem.startswith('_'):
            import_module(file)
