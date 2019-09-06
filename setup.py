import re

from setuptools import setup, find_packages

# load version from module (without loading the whole module)
with open('face_engine/__init__.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)


setup(
    name='face-engine',
    version=version,
    description='Face recognition engine',
    long_description=__doc__,
    license='Apache 2.0',
    author='Daniyar Kussainov',
    author_email='ohw0sseug@gmail.com',
    url="https://github.com/guesswh0/face_engine",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'scikit-image',
        'scikit-learn',
        'dlib>=19.17',
    ],
    entry_points={
        'console_scripts': [
            'fetch_models = fetching.fetch_models:fetch',
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
    ],
    python_requires='>=3.6'
)
