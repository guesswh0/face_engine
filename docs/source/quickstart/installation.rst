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
