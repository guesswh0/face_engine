Installation
============

It is distributed on `PyPi`_, and can be installed with pip:

.. code-block:: console

    $ pip install face-engine

.. note::
   FaceEngine is supported only on Python 3.11 and above.

To install the default insightface backend models use:

.. code-block:: console

    $ pip install face-engine[insightface]

.. note::
   **GPU usage**: the ``[insightface]`` extra installs the CPU build of
   onnxruntime. To run the insightface models on an NVIDIA GPU replace it
   with ``pip install onnxruntime-gpu`` (requires CUDA/cuDNN); no code
   changes are needed — the models pick the CUDA execution provider
   automatically when it is available. On CPU-only machines onnxruntime
   logs a harmless ``CUDAExecutionProvider is not in available provider
   names`` warning and falls back to CPU. See the `onnxruntime execution
   providers`_ documentation for details.

.. _onnxruntime execution providers: https://onnxruntime.ai/docs/execution-providers/

To install the legacy dlib backend either run ``pip install dlib``
(compiles from source, requires cmake) or install prebuilt wheels with:

.. code-block:: console

    $ pip install dlib-bin

.. _PyPi: https://pypi.org/project/face-engine/

Manually
--------

Download or pull sources from project `repository`_ and run:

.. code-block:: console

   $ pip install .

.. _repository: https://github.com/guesswh0/face_engine.git
