# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FaceEngine is a lightweight Python library (published on PyPI as `face-engine`) providing a pluggable interface for face recognition tasks. Python >= 3.11. Package version is dynamic, read from `__version__` in `face_engine/__init__.py`.

## Commands

```bash
# install for development (pulls the insightface backend)
pip install -e ".[dev]"

# optional legacy dlib backend: official pypi 'dlib' compiles from source
# (needs cmake); 'dlib-bin' ships prebuilt wheels
pip install dlib-bin

# run all tests
pytest

# run a single test file / test
pytest tests/test_face_engine.py
pytest tests/test_face_engine.py::TestFaceEngine::test_getters

# antelopev2 model tests (407 MB download) are opt-in
FACE_ENGINE_TEST_ANTELOPE=1 pytest tests/test_face_engine_models.py

# lint / format
ruff check .
black .

# build docs (sphinx)
cd docs && make html
```

Note on tests: `tests/__init__.py` downloads test images and datasets from GitHub at import time (into the resource cache, see below), so the first test run needs network access. Tests are skipped or switched via `@unittest.skipUnless(dlib, ...)` / `skipUnless(insightface, ...)` decorators, forming a 3-way backend matrix: insightface installed / only dlib / no backends. All three combinations must pass.

## Architecture

### Plugin model registry (PEP 487)

The core design is a plugin system built on `__init_subclass__` subclass registration in `face_engine/models/__init__.py`:

- Three abstract model interfaces: `Detector` (find faces → bounding boxes), `Embedder` (bounding boxes → embedding vectors), `Estimator` (fit/predict on embeddings).
- Any subclass declared with a `name` class keyword (e.g. `class SCRFDDetector(Detector, name="scrfd")`) is automatically registered in the module-level `_models` dict at import time. Extra registry keys can be added with the `aliases` class keyword (e.g. `retina_face` is a deprecated alias of `scrfd`); the `name` descriptor keeps the canonical name, and getters return it even when set via an alias. `Embedder` subclasses also take a `dim` keyword (embedding dimension) — subclasses of a registered embedder must re-pass `dim` or `embedding_dim` resets to None.
- `import_package(__file__)` at the bottom of `models/__init__.py` auto-imports every non-underscore module in `face_engine/models/`, so dropping a new module in that directory registers its models. `ImportError` is silently swallowed for the optional built-in modules (`dlib_models`, `insightface_models`) so the library works without those dependencies.
- To add custom models outside the package, users simply implement the interfaces and import their module — registration is a side effect of class creation.

### FaceEngine core (`face_engine/core.py`)

`FaceEngine` composes the three models and resolves them by name string via property setters (`engine.detector = "scrfd"` instantiates the registered class). Empty names resolve through installed-backend fallback chains (`_DETECTOR_DEFAULTS` etc.: insightface `scrfd`/`arcface` → dlib `hog`/`resnet` → `abstract_*`); explicit unknown names warn and fall back to the `abstract_*` no-op models.

Pipeline: `make_prediction()` = `find_faces()` (detector) → `compute_embeddings()` (embedder) → `predict()` (estimator). Detectors return `(bounding_boxes, extra)` where `extra` is a model-dependent dict passed through to the embedder as kwargs (e.g. insightface's `kpss` keypoints — the SCRFD detectors and ArcFace embedders are coupled through it). `find_faces(limit=N)` keeps the N largest faces (stable sort, so equal-sized dlib boxes keep detector order) and filters all `extra` values by the same indices.

### Persistence (no pickle)

`engine.save(filename)` writes a JSON document (`format: "face-engine"`) holding model names, counters, and any extra JSON-serializable attributes; `load_engine(filename)` re-instantiates models from names. Estimator training state is persisted separately by the estimator itself in the same directory as the engine file — `BasicEstimator` uses `basic.estimator.npz` (loaded with `allow_pickle=False`) plus `basic.estimator.json` for class names — and is only saved/loaded when `n_samples > 0`. Loading pre-3.0 pickle files raises `RuntimeError` with a re-fit message; pickle must not be reintroduced (arbitrary code execution on load).

### Resource fetching (`face_engine/fetching.py`)

Pre-trained model files and test data are downloaded on first use to `RESOURCES` (`platformdirs.user_cache_dir("face_engine")`) and cached; archives (`.bz2`, `.zip`, etc.) are unpacked automatically. Downloads are streamed with a timeout and verified against pinned SHA-256 checksums in `KNOWN_HASHES` (keyed by origin filename — add an entry when adding a model file); non-http(s) URL schemes are rejected. Model constructors call `fetch_file()`/`_fetch_pack()` in `__init__`, so instantiating e.g. `SCRFDDetector` triggers a download. insightface packs come from the GitHub v0.7 release assets (`storage.insightface.ai` is dead); `antelopev2.zip` extracts into a nested `antelopev2/` directory, which `_fetch_pack` handles.

### Licensing caveat

The library is Apache-2.0, but the insightface model pack weights (buffalo_l, antelopev2) are licensed for **non-commercial research use only**. Keep that documented in README when touching model docs.

### Layout notes

- `extra/` — dataset preparation CLI scripts; excluded from the built package (as is `tests/`). Still uses pickle for its own dataset files (dev/test tooling only — intentionally out of scope).
- Bounding box format everywhere is `(left, upper, right, lower)`.
- Images are RGB numpy arrays; `face_engine.tools.imread` accepts file paths, bytes, file objects, and URLs (URL downloads are LRU-cached).
