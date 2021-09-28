FaceEngine
==========

FaceEngine is a lightweight python library that provides an easy interface to
work with **face recognition** tasks.

.. code-block:: python

    >>> from face_engine import FaceEngine
    >>> engine = FaceEngine()
    >>> engine.fit(['bubbles1.jpg', 'drive.jpg'], [1, 2])
    >>> engine.make_prediction('bubbles2.jpg')
    ([(270, 75, 406, 211)], [1])


Installation
------------

It is distributed on `PyPi`_, and can be installed with pip:

.. code-block:: console

    $ pip install face-engine

FaceEngine is supported only on Python 3.6 and above.

.. _PyPi: https://pypi.org/project/face-engine/
 
Models
------

FaceEngine is built on top of three model interfaces ``Detector``, ``Embedder``
and ``Estimator`` (see `models`_), and leans on user provided implementations
of these models.

Installation provides optional dlib models.

These implementations are using `dlib python api`_ and dlib provided
pre-trained model `files`_.

.. note::
   FaceEngine installation is not installing dlib by default.
   To install it, either run ``pip install dlib`` (requires cmake) or
   follow `build instructions`_.

   Dlib models implementations are used to show how to work with FaceEngine.
   Questions and issues according to these models accuracy and performance
   please address to dlib.

To work with your own custom models you have to implement required
`models`_ and import it. FaceEngine models are used to register all inheriting
imported subclasses (subclass registration `PEP 487`_).

For more information read the full `documentation`_.

.. _models: https://github.com/guesswh0/face_engine/blob/master/face_engine/models/__init__.py
.. _dlib python api: http://dlib.net/python/index.html
.. _files: http://dlib.net/files/
.. _build instructions: http://dlib.net/compile.html
.. _PEP 487: https://www.python.org/dev/peps/pep-0487/
.. _documentation: https://face-engine.readthedocs.io/en/latest/
