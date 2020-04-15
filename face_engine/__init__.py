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

__all__ = ['__version__', 'logger', 'RESOURCES', 'FaceEngine', 'models']

__version__ = '1.3.0'

import logging
import os

logger = logging.getLogger(__name__)

RESOURCES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'resources'))

from face_engine.core import FaceEngine, models
