Recognition point
=================


Facial recognition like most of current (digital) recognition techniques is
based on these steps:

* ``Face detection`` (:class:`detector <~face_engine.models.Detector>`) - is an
  essential step as it detects and locates human faces in images and videos.
  At the moment there are dozens of techniques and solutions starting from
  Haar-like cascades to deep neural networks. The point is to answer a
  question: is there a face in this picture? Finally, face detection determines
  the presence, location of the face as a bounding box and (possibly)
  the confidence score of how likely is it a human face.

* ``Face digitization`` (:class:`embedder <~face_engine.models.Embedder>`) - so
  called technique that transforms analogue information (a picture of face)
  into a set of digital information based on unique person's facial features,
  to be more precise - calculates the facial features as n-dimensional embedding
  vectors to uniquely locate it in n-dimensional space.

* ``Making prediction`` (:class:`estimator <~face_engine.models.Estimator>`) -
  the last step of facial recognition accumulates previously calculated data and
  compares it with already existing ones. Comparing as 1to1 (verification task)
  or 1toN (identification task) does not differ from other machine learning
  tasks, the key is to make some prediction on given (digital) data.