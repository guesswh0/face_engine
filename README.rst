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

    $ pip install face-engine[insightface]

FaceEngine is supported only on Python 3.11 and above.

.. _PyPi: https://pypi.org/project/face-engine/

Models
------

FaceEngine is built on top of four model interfaces ``Detector``, ``Embedder``,
``Estimator`` and ``Antispoof`` (see `models`_), and leans on user provided
implementations of these models.

The default backend is `insightface`_ (the ``[insightface]`` extra), with
these bundled model implementations:

======================= ========== ============ ==========================
name                    role       model pack   notes
======================= ========== ============ ==========================
``scrfd``               detector   buffalo_l    default; deprecated alias
                                                ``retina_face``
``arcface``             embedder   buffalo_l    default, 512-d
``scrfd_antelopev2``    detector   antelopev2   opt-in
``arcface_antelopev2``  embedder   antelopev2   strongest insightface
                                                embedder, 512-d
``minifasnet``          antispoof  --           passive anti-spoofing,
                                                opt-in
======================= ========== ============ ==========================

.. note::
   The ``[insightface]`` extra installs the CPU build of onnxruntime.
   For NVIDIA GPU inference replace it with the GPU build — both
   packages provide the same ``onnxruntime`` module, so the CPU build
   must be removed first::

       pip uninstall onnxruntime && pip install onnxruntime-gpu

   The models use the CUDA execution provider automatically when
   available.

Legacy `dlib python api`_ models (``hog``, ``mmod`` detectors and ``resnet``
embedder with dlib pre-trained model `files`_) are kept as an optional
fallback backend used when insightface is not installed.

.. note::
   FaceEngine installation is not installing dlib by default.
   To install it, either run ``pip install dlib`` (requires cmake),
   install prebuilt wheels with ``pip install dlib-bin``, or follow
   `build instructions`_.

To work with your own custom models you have to implement required
`models`_ and import it. FaceEngine models are used to register all inheriting
imported subclasses (subclass registration `PEP 487`_).

Face anti-spoofing
------------------

Since 3.1 the engine has an opt-in liveness (presentation attack detection)
step powered by the ``Antispoof`` model interface:

.. code-block:: python

    >>> engine = FaceEngine(antispoof="minifasnet")
    >>> engine.check_liveness('bubbles1.jpg')
    array([0.971], dtype=float32)

The bundled ``minifasnet`` model is an ensemble of the two released
`Silent-Face-Anti-Spoofing`_ MiniFASNet models (requires ``onnxruntime``,
already present with the ``[insightface]`` extra). It is effective against
printed photos and basic screen replays; it is **not** a certified
(ISO/IEC 30107-3) liveness solution.

Face verification
-----------------

Since 3.2 the engine has a model-agnostic 1:1 verification primitive —
cosine similarity between two embeddings of the same embedder:

.. code-block:: python

    >>> bbs, extra = engine.find_faces(image, limit=1)
    >>> source = engine.compute_embeddings(image, bbs, **extra)[0]
    >>> engine.compare(source, target)
    0.83

``compare`` returns the raw score in ``[-1, 1]``; accept/reject
thresholding is left to the caller and should be calibrated per embedder.

Model weights licensing
-----------------------

The library code is Apache-2.0, but the downloaded pre-trained model weights
come with their own terms:

* insightface model packs (``buffalo_l``, ``antelopev2``) are available for
  **non-commercial research purposes only** (see `insightface`_).
* dlib model files have their own terms, see `dlib-models`_.
* ``minifasnet`` model weights are **Apache-2.0** (usable commercially) —
  ONNX exports of the `Silent-Face-Anti-Spoofing`_ checkpoints, reproducible
  with ``extra/export_minifasnet.py``.

Changes in 3.2
--------------

* Unknown explicit model names now raise ``ModelNotFoundError`` instead of
  warning and falling back to the abstract no-op models. This applies to the
  ``FaceEngine`` constructor, the model property setters, and
  ``load_engine``. Empty names keep the installed-backend fallback chains.
* New ``engine.compare(source, target)`` — cosine similarity between two
  embeddings for 1:1 verification (raw score, no thresholding).
* ``tools.imread`` now reads raw ``bytes`` image content, as its
  documentation always claimed.

Breaking changes in 3.0
-----------------------

* Python >= 3.11 is required.
* Pickle persistence was removed for security reasons: engines are saved as
  JSON (``engine.save('engine.json')``) and estimator state as ``.npz`` +
  ``.json`` files. Engines saved with face-engine < 3.0 cannot be loaded —
  re-fit and save again.
* With insightface installed the default models are ``scrfd``/``arcface``
  (previously dlib ``hog``/``resnet``).
* The ``retina_face`` detector was renamed to ``scrfd`` (the actual model
  architecture); the old name is kept as a deprecated alias.
* Model downloads are verified against pinned SHA-256 checksums.

For more information read the full `documentation`_.

.. _models: https://github.com/guesswh0/face_engine/blob/master/face_engine/models/__init__.py
.. _insightface: https://github.com/deepinsight/insightface
.. _Silent-Face-Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
.. _dlib python api: http://dlib.net/python/index.html
.. _files: https://github.com/davisking/dlib-models
.. _dlib-models: https://github.com/davisking/dlib-models
.. _build instructions: http://dlib.net/compile.html
.. _PEP 487: https://www.python.org/dev/peps/pep-0487/
.. _documentation: https://face-engine.readthedocs.io/en/latest/
