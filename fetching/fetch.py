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

import fetching
from face_engine import RESOURCES


def fetch_images():
    """Fetch testing images"""

    for fname in [
        "images/drive.jpg",
        "images/family.jpg"
    ]:
        fetching.get_remote_file(fname, RESOURCES)


def fetch_models():
    """Fetch default dlib models for face recognition and pip install dlib"""

    try:
        import dlib
    except ModuleNotFoundError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'dlib'])

    extract_dir = os.path.join(RESOURCES, 'data')
    url_root = "http://dlib.net/files/"
    # Load dlib models
    for name in [
        "mmod_human_face_detector.dat.bz2",
        "dlib_face_recognition_resnet_model_v1.dat.bz2",
        "shape_predictor_5_face_landmarks.dat.bz2",

    ]:
        filename = fetching.get_remote_file(name, extract_dir, url_root)
        if filename:
            fetching.unpack_archive(name, extract_dir)
            os.remove(filename)


def fetch_all():
    fetch_models()
    fetch_images()


if __name__ == '__main__':
    fetch_all()
