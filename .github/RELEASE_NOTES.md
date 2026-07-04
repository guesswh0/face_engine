# face-engine 3.1.0

Face anti-spoofing (liveness) support.

## Highlights

- **New `Antispoof` model interface** — the fourth model interface alongside
  `Detector`/`Embedder`/`Estimator`, registered through the same plugin
  system. `FaceEngine` gains an opt-in `antispoof` slot and a
  `check_liveness(image, bounding_boxes=None)` method returning live-face
  probabilities.
- **New `minifasnet` model** — passive presentation-attack detection,
  an ensemble of the two released
  [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
  MiniFASNet models. Requires only `onnxruntime` + numpy (no extra
  dependencies with the `[insightface]` extra installed).

  ```python
  engine = FaceEngine(antispoof="minifasnet")
  engine.check_liveness("selfie.jpg")  # array([0.971], dtype=float32)
  ```

- The `minifasnet` weights are **Apache-2.0 — usable in commercial
  products** (unlike the insightface packs). They are ONNX exports of the
  official checkpoints, reproducible with `extra/export_minifasnet.py`
  (pinned upstream commit + SHA-256-verified checkpoints), downloaded with
  pinned SHA-256 checksums like every other model file.

## Notes

- Effective against printed photos and basic screen replays; **not** a
  certified (ISO/IEC 30107-3) liveness solution.
- Engines saved by 3.0 load fine (the antispoof slot defaults to the
  abstract no-op model); engines saved by 3.1 store the antispoof model
  name.

## Model weights licensing

The library is Apache-2.0. The insightface model pack weights (buffalo_l,
antelopev2) remain licensed for **non-commercial research purposes only**;
the `minifasnet` weights are Apache-2.0.
