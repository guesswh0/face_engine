# face-engine 3.3.0

The insightface models now see the channel order they were trained on.

## Highlights

- **insightface models run on BGR, as upstream does.** `FaceEngine`
  images are RGB. insightface's ONNX wrappers (`SCRFD`, `ArcFaceONNX`)
  swap channels on the assumption that the input is OpenCV's BGR, so
  through 3.2 the `scrfd` detectors and the `arcface` embedders received
  swapped channels. The engine now converts to BGR before the model,
  exactly as insightface's own pipeline does, and plugins that inherit
  `ArcFaceEmbedder` get the same correction. Measured on LFW (10-fold
  pairs, identical detections): `arcface` moves from EER 0.0113 to
  0.0083 and from TAR 0.9870 to 0.9907 at FMR = 1e-4; an Apache-licensed
  arcface-family plugin model moves from TAR 0.8870 to 0.9627 at the
  same point.

## Behaviour changes

- **Embeddings from the `arcface` family are a new space.** Vectors
  computed with 3.2 and earlier are not comparable to 3.3 vectors of the
  same face at calibrated thresholds: on one arcface-family model the
  cosine between the two, crop for crop, averages 0.88 over 121k faces
  and drops below 0.5 in the tail. Re-embed stored enrollments, re-fit
  any `BasicEstimator` trained on them, and re-calibrate accept/reject
  thresholds per embedder.
- `scrfd` detections shift slightly, since the detector now sees the
  right channel order.
- `minifasnet`, the dlib models, and plugins that convert to BGR
  themselves are unaffected.

## Model weights licensing

The library is Apache-2.0. The insightface model pack weights (buffalo_l,
antelopev2) remain licensed for **non-commercial research purposes only**;
the `minifasnet` weights are Apache-2.0.
