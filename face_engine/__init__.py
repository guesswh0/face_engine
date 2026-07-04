"""
Face Recognition Engine
"""

__all__ = ["logger", "RESOURCES", "FaceEngine", "_models", "load_engine"]
__version__ = "3.0.0"
__author__ = "Daniyar Kussainov"
__license__ = "Apache License, Version 2.0"
__copyright__ = "Copyright 2019-2026 Daniyar Kussainov"

import logging

logger = logging.getLogger(__name__)

from face_engine.fetching import RESOURCES  # noqa: E402
from face_engine.core import FaceEngine, _models, load_engine  # noqa: E402
