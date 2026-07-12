# face-engine 3.2.0

Strict model resolution, a 1:1 verification primitive, and an `imread`
bytes fix.

## Highlights

- **Strict model resolution** — an explicit model name that is not in the
  registry now raises the new `ModelNotFoundError` instead of warning and
  silently falling back to the abstract no-op models. This applies to the
  `FaceEngine` constructor, the model property setters, and `load_engine`.
  Empty names keep the installed-backend fallback chains (insightface →
  dlib → abstract). For the in-tree optional models
  (`scrfd`/`arcface`/`hog`/`resnet`/`mmod`/`minifasnet`) the error names
  the missing backend dependency instead of suggesting a module import.
- **New `FaceEngine.compare(source, target)`** — cosine similarity between
  two embeddings of the same embedder, the 1:1 verification primitive:

  ```python
  bbs, extra = engine.find_faces(image, limit=1)
  source = engine.compute_embeddings(image, bbs, **extra)[0]
  engine.compare(source, target)  # 0.83
  ```

  Returns the raw score in `[-1, 1]` (zero-norm inputs score `0.0`);
  accept/reject thresholding is left to the caller and should be
  calibrated per embedder.
- **`tools.imread` reads raw `bytes`** — as its documentation always
  claimed. Previously a bare `bytes` object fell through to
  `PIL.Image.open`, which treats it as a file path and fails on any real
  image content.

## Breaking changes

- Code that relied on unknown model names degrading to abstract no-op
  models must either use registered names or catch `ModelNotFoundError`.
  A saved engine file naming an unavailable model (e.g. a plugin module
  not imported at load time) now raises on `load_engine` instead of
  loading a silently non-functional engine.
- `bytes` passed to `imread` is now decoded as image content, never
  treated as a filesystem path.

## Model weights licensing

The library is Apache-2.0. The insightface model pack weights (buffalo_l,
antelopev2) remain licensed for **non-commercial research purposes only**;
the `minifasnet` weights are Apache-2.0.
