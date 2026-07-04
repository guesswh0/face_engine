Models
======

FaceEngine is built on top of four model interfaces
:class:`~face_engine.models.Detector`, :class:`~face_engine.models.Embedder`,
:class:`~face_engine.models.Estimator` and
:class:`~face_engine.models.Antispoof`, and leans on user provided
implementations of these models.


.. _default-models:

Default models
--------------

The default backend is `insightface`_ (installed with
``pip install face-engine[insightface]``), providing the
:ref:`insightface models <insightface-models>`:

======================= ========== ============ ===========================
name                    role       model pack   notes
======================= ========== ============ ===========================
``scrfd``               detector   buffalo_l    default; deprecated alias
                                                ``retina_face``
``arcface``             embedder   buffalo_l    default, 512-d
``scrfd_antelopev2``    detector   antelopev2   opt-in
``arcface_antelopev2``  embedder   antelopev2   strongest insightface
                                                embedder, 512-d
======================= ========== ============ ===========================

.. warning::
   The insightface pre-trained model weights (buffalo_l, antelopev2 packs)
   are licensed for **non-commercial research purposes only**. The library's
   Apache-2.0 license does not extend to the downloaded model files.

If insightface is not installed, the engine falls back to the legacy
:ref:`dlib models <dlib-models>` (``hog`` detector, ``resnet`` embedder).

Face anti-spoofing is opt-in via the ``minifasnet``
:ref:`antispoof model <antispoof-models>` (requires ``onnxruntime``,
already present with the ``[insightface]`` extra):

.. code-block:: python

   engine = FaceEngine(antispoof="minifasnet")
   scores = engine.check_liveness('selfie.jpg')

.. note::
   The ``minifasnet`` weights are Apache-2.0 (usable commercially) — ONNX
   exports of the released `Silent-Face-Anti-Spoofing`_ checkpoints,
   reproducible with ``extra/export_minifasnet.py``. The model is effective
   against printed photos and basic screen replays, but is not a certified
   (ISO/IEC 30107-3) liveness solution.

.. note::
   FaceEngine installation is not installing dlib by default.
   To install it, either run ``pip install dlib`` (requires cmake),
   install prebuilt wheels with ``pip install dlib-bin``, or
   follow `build instructions`_.

   Dlib model files come with their own terms, see `files`_.

At the moment there is:
    * :class:`~face_engine.models.insightface_models.SCRFDDetector`
    * :class:`~face_engine.models.insightface_models.SCRFDAntelopeV2Detector`
    * :class:`~face_engine.models.insightface_models.ArcFaceEmbedder`
    * :class:`~face_engine.models.insightface_models.ArcFaceAntelopeV2Embedder`
    * :class:`~face_engine.models.dlib_models.HOGDetector`
    * :class:`~face_engine.models.dlib_models.MMODDetector`
    * :class:`~face_engine.models.dlib_models.ResNetEmbedder`
    * :class:`~face_engine.models.minifasnet.MiniFASNetAntispoof`
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


.. _insightface: https://github.com/deepinsight/insightface
.. _Silent-Face-Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
.. _dlib python api: http://dlib.net/python/index.html
.. _files: https://github.com/davisking/dlib-models
.. _build instructions: http://dlib.net/compile.html
.. _PEP 487: https://www.python.org/dev/peps/pep-0487/
