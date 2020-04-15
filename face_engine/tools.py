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

from importlib import util
from pathlib import Path

from . import logger


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
