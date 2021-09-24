import os
import re

from setuptools import setup, find_packages


def get_version(package):
    """ Read version from package (without loading the whole package)"""

    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


def long_description():
    with open('README.rst', encoding='utf-8') as f:
        return f.read()


setup(
    name='face-engine',
    version=get_version('face_engine'),
    description='Face Recognition Engine',
    long_description=long_description(),
    long_description_content_type='text/x-rst',
    license='Apache License, Version 2.0',
    author='Daniyar Kussainov',
    author_email='ohw0sseug@gmail.com',
    url="https://github.com/guesswh0/face_engine",
    packages=find_packages(exclude=['tests', 'extra']),
    install_requires=[
        'numpy>=1.18.0,<2.0',
        'pillow>=8.0.0,<9.0',
        'tqdm>=4.44.0,<5.0'
    ],
    extras_require={
        'dev': [
            'dlib',
            'opencv-python-headless',
            'sphinx',
            'sphinx-rtd-theme'
        ]
    },
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Operating System :: OS Independent',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.6'
)
