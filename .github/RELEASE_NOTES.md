# face-engine 3.0.0

Modernization release, 7 years after the first one. Validated on macOS/arm64, Linux/x86_64 (Python 3.11 & 3.12) and Linux/CUDA (Tesla T4).

## Highlights

- **InsightFace is the new default backend** (`pip install face-engine[insightface]`, insightface >= 1.0.1): SCRFD detector + ArcFace embedder from the `buffalo_l` pack, plus the stronger `antelopev2` pack as opt-in (`scrfd_antelopev2`, `arcface_antelopev2`). Runs on GPU automatically with `onnxruntime-gpu` (see README).
- **Pickle persistence removed** (security): engines save as JSON, estimator state as `.npz` + `.json`. Engines saved with < 3.0 cannot be loaded — re-fit and save again.
- **Model downloads are SHA-256 verified**, streamed with a timeout, and fetched from the official insightface GitHub release assets (`storage.insightface.ai` is gone).
- **Dependency floors clear known CVEs**: pillow >= 12.2, tqdm >= 4.66.3, numpy >= 1.26.
- Fixed `find_faces(limit=N)`: it now really keeps the N largest faces and filters detector extras (`det_scores`, `kpss`) consistently; previously extras were mis-indexed.

## Breaking changes

- Python >= 3.11 is required.
- With insightface installed the default models are `scrfd`/`arcface` (previously dlib `hog`/`resnet`). dlib remains supported as an optional legacy backend (`pip install dlib-bin` for prebuilt wheels).
- The `retina_face` detector was renamed to `scrfd` (its actual architecture); the old name still works as a deprecated alias.
- Saved engines from 2.x (pickle) are not loadable; loading raises a clear error.

## Model weights licensing

The library is Apache-2.0, but the insightface model pack weights (buffalo_l, antelopev2) are licensed for **non-commercial research purposes only**.
