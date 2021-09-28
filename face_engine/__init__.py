"""
Face Recognition Engine
"""

__all__ = ['logger', 'RESOURCES', 'FaceEngine', '_models', 'load_engine']
__version__ = '2.1.0'
__author__ = 'Daniyar Kussainov'
__license__ = 'Apache License, Version 2.0'
__copyright__ = 'Copyright 2019-2020 Daniyar Kussainov'

import logging
import os

logger = logging.getLogger(__name__)

RESOURCES = os.path.expanduser('~/.face_engine')

from face_engine.core import FaceEngine, _models, load_engine
