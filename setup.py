import re

from setuptools import setup, find_packages

# load version from module (without loading the whole module)
with open('face_engine/__init__.py') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='face-engine',
    version=version,
    description='Face recognition engine',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    author='Daniyar Kussainov',
    author_email='ohw0sseug@gmail.com',
    url="https://github.com/guesswh0/face_engine",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy~=1.18.0',
        'pillow~=7.0.0',
        'tqdm~=4.44.0'
    ],
    extras_require={
        'dev': [
            'scikit-learn',
            'dlib~=19.0',
            'opencv-python-headless~=3.4'
        ]
    },
    entry_points={
        'console_scripts':
            [
                'fetch_images=face_engine.fetching:fetch_images',
                'fetch_models=face_engine.fetching:fetch_models'
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
