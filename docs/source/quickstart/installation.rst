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
   onnxruntime. To run the insightface models on an NVIDIA GPU replace
   it with the GPU build (requires CUDA/cuDNN):

   .. code-block:: console

       $ pip uninstall onnxruntime && pip install onnxruntime-gpu

   Both packages provide the same ``onnxruntime`` module, so installing
   ``onnxruntime-gpu`` alongside the CPU build (which the
   ``[insightface]`` extra always pulls in) leaves whichever installed
   last — the CPU build must be removed explicitly. Pip will warn that
   ``insightface requires onnxruntime, which is not installed`` — this
   is harmless, ``onnxruntime-gpu`` provides the same module. No code
   changes are needed after that: the models pick the CUDA execution
   provider automatically when it is available, and fall back to CPU
   otherwise. See the `onnxruntime execution providers`_ documentation
   for details.

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
