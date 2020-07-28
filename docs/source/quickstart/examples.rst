Usage examples
==============

These examples are using :ref:`default models <default-models>`, therefore
before trying it be sure that you have dlib installed. Also in some examples
for simplicity purposes, :ref:`exceptions <exceptions>` are not handled.

Getting working instances
-------------------------

Working with pre-defined default models.


.. code-block:: python

    >>> from face_engine import FaceEngine
    >>> engine = FaceEngine()
    >>> engine.detector
    'hog'
    >>> engine.embedder
    'resnet'
    >>> engine.estimator
    'basic'

Notice that if dlib is not installed ``detector`` and ``embedder`` models will
be abstract.

.. code-block:: python

    >>> from face_engine import FaceEngine
    >>> engine = FaceEngine()
    >>> engine.detector
    'abstract_detector'
    >>> engine.embedder
    'abstract_embedder'
    >>> engine.estimator
    'basic'

To work with your own custom detector use:

.. code-block:: python

    >>> from my_custom_models import my_custom_detector
    >>> engine = FaceEngine(detector='custom_detector')

or use corresponding setter method:

.. code-block:: python

    >>> from face_engine import FaceEngine
    >>> from my_custom_models import my_custom_detector
    >>> engine = FaceEngine()
    >>> engine.detector = 'custom_detector'


Face detection
--------------

Find all faces in the image with corresponding confidence scores.

.. code-block:: python

    from face_engine import FaceEngine
    engine = FaceEngine()
    confidences, bounding_boxes = engine.find_faces('bubbles1.jpg')


Find largest face bounding box in the image with corresponding
confidence score.

.. code-block:: python

    from face_engine import FaceEngine
    engine = FaceEngine()
    confidence, bounding_box = engine.find_face('bubbles1.jpg')


Face recognition
----------------

These examples are using :func:`~face_engine.tools.imread` function to read
image as :class:`~numpy.uint8` array.

Extract facial embedding vectors from the image.

.. code-block:: python

    from face_engine import FaceEngine, tools
    engine = FaceEngine()
    image = tools.imread('bubbles1.jpg)
    bbs = engine.find_faces(image)[1]
    embeddings = engine.compute_embeddings(image, bbs)


Predict class name by given face image.

.. note:: model has to be fitted before making any predictions.

.. code-block:: python

    from face_engine import FaceEngine, tools
    engine = FaceEngine()
    engine.fit(['bubbles1.jpg', 'drive.jpg'], [1, 2])

    image = tools.imread('bubbles2.jpg')
    bbs = engine.find_faces(image)[1]
    embeddings = engine.compute_embeddings(image, bbs)
    scores, class_names = engine.predict(embeddings)

Make (lazy) prediction to find out class names and bounding boxes in one call.

.. code-block:: python

    from face_engine import FaceEngine
    engine = FaceEngine()
    engine.fit(['bubbles1.jpg', 'drive.jpg'], [1, 2])
    class_names, bounding_boxes = engine.make_prediction('bubbles2.jpg')


Face comparison
---------------

Compare a face in the source image with each face in the
target image, to find out the most similar one.

.. code-block:: python

    from face_engine import FaceEngine, tools
    engine = FaceEngine()
    score, bounding_box = engine.compare_faces('bubbles1.jpg', 'bubbles2.jpg')


Persistence
-----------

Save engine state to file:

.. code-block:: python

    >>> from face_engine import FaceEngine
    >>> engine = FaceEngine()
    >>> engine.fit(['bubbles1.jpg', 'drive.jpg'], [1, 2])
    >>> engine.save('engine.p')


Load engine state from file:

.. code-block:: python

    >>> from face_engine import load_engine
    >>> engine = load_engine('engine.p')
    >>> engine.make_prediction('bubbles2.jpg')
    >>> ([(270, 75, 406, 211)], [1])

Application examples
--------------------

These examples use `opencv`_ to read and visualize image data, so you may
need to install it before.

.. _opencv: https://pypi.org/project/opencv-python/

Live face detection
'''''''''''''''''''

.. code-block:: python

    import cv2
    from face_engine import FaceEngine
    from face_engine.exceptions import FaceNotFoundError

    engine = FaceEngine()
    cap = cv2.VideoCapture(0)

    while True:
        frame = cap.read()[1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            bbs = engine.find_faces(rgb_frame)[1]
            for bb in bbs:
                cv2.rectangle(frame, bb[:2], bb[2:], (0, 255, 0), 1)
        except FaceNotFoundError:
            pass
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


Live face recognition
'''''''''''''''''''''

This example use :meth:`~face_engine.FaceEngine.make_prediction`, to find
out only class names without prediction scores. To get prediction scores use
:meth:`~face_engine.FaceEngine.predict` instead.

.. code-block:: python

    import cv2
    from face_engine import load_engine
    from face_engine.exceptions import FaceNotFoundError

    # assume that engine is saved before
    engine = load_engine('engine.p')
    cap = cv2.VideoCapture(0)

    while True:
        frame = cap.read()[1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            bbs, names = engine.make_prediction(rgb_frame)
        except FaceNotFoundError:
            pass # pass drawing
        else:
            # draw bounding boxes and predicted names
            for bb, name in zip(bbs, names):
                cv2.rectangle(frame, bb[:2], bb[2:], (0, 255, 0), 1)
                cv2.putText(frame, name, (bb[0], bb[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


