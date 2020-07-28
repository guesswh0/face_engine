Models
======

FaceEngine is built on top of three model interfaces
:class:`~face_engine.models.Detector`, :class:`~face_engine.models.Embedder`
and :class:`~face_engine.models.Estimator`, and leans on user provided
implementations of these models.


.. _default-models:

Default models
--------------

Installation provides optional :ref:`dlib models <dlib-models>`.

These implementations are using `dlib python api`_ and dlib provided pre-trained
model `files`_, which should be fetched manually:

.. code-block:: console

    $ fetch_models

.. note::
   FaceEngine installation is not installing dlib by default.
   To install it, either run ``pip install dlib`` (requires cmake) or
   follow `build instructions`_.

   Dlib models implementations are used to show how to work with FaceEngine.
   Questions and issues according to these models accuracy and performance
   please address to dlib.

At the moment there is:
    * :class:`~face_engine.models.dlib_models.HOGDetector`
    * :class:`~face_engine.models.dlib_models.MMODDetector`
    * :class:`~face_engine.models.dlib_models.ResNetEmbedder`
    * :class:`~face_engine.models.basic_estimator.BasicEstimator`


Custom models
-------------

To work with your own custom models you have to implement required
:ref:`models <models>` and import it with either directly importing your model
or adding it to ``PYTHONPATH`` environment variable or using appropriate
convenient function from :mod:`face_engine.tools`. This will register your model
class object itself in ``face_engine._models`` dictionary, from where it becomes
visible.

FaceEngine models are used to register all inheriting imported subclasses
(subclass registration `PEP 487`_).


For example to initialize FaceEngine with your own custom detector use
appropriate keyword argument with model ``name``:

.. code-block:: python

   from face_engine import FaceEngine
   from my_custom_models import my_custom_detector
   engine = FaceEngine(detector='custom_detector')

or use corresponding setter method with model ``name``:

.. code-block:: python

   from face_engine import FaceEngine
   from my_custom_models import my_custom_detector
   engine = FaceEngine()
   engine.detector = 'custom_detector'


.. _dlib models: https://github.com/guesswh0/face_engine/blob/master/face_engine/models/dlib_models.py
.. _dlib python api: http://dlib.net/python/index.html
.. _files: http://dlib.net/files/
.. _build instructions: http://dlib.net/compile.html
.. _PEP 487: https://www.python.org/dev/peps/pep-0487/
