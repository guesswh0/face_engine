#!/usr/bin/env python3
"""One-time dev script: export MiniFASNet anti-spoofing checkpoints to ONNX.

Reproducible provenance chain for the ``minifasnet`` face-engine model:

1. Clones minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0) at a
   pinned commit for the reference architecture code and the released
   checkpoints, verifying the checkpoint SHA-256s.
2. Exports both released 80x80 models to ONNX (logits output, dynamic
   batch, opset 17).
3. Validates ONNX == PyTorch outputs on random 0-255 inputs.
4. Prints the SHA-256 of the produced files to pin in
   ``face_engine.fetching.KNOWN_HASHES``.

The resulting files are published as face-engine GitHub release assets;
this script exists so anyone can reproduce them from upstream.

Requires (dev-only, not face-engine deps): torch, onnx, onnxruntime.

    $ python export_minifasnet.py [output_dir]
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_URL = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
# master as of 2020-08-05, the repo is unmaintained since
REPO_COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"

# released checkpoints: file -> (model_type, conv6_kernel, sha256)
CHECKPOINTS = {
    "2.7_80x80_MiniFASNetV2.pth": (
        "MiniFASNetV2",
        (5, 5),
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    ),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (
        "MiniFASNetV1SE",
        (5, 5),
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
    ),
}

OPSET = 17


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clone_upstream(target):
    subprocess.run(["git", "clone", REPO_URL, str(target)], check=True)
    subprocess.run(["git", "checkout", REPO_COMMIT], cwd=target, check=True)


def export(upstream, output_dir):
    import torch
    from src.model_lib import MiniFASNet

    model_mapping = {
        "MiniFASNetV2": MiniFASNet.MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNet.MiniFASNetV1SE,
    }

    for filename, (model_type, kernel, checksum) in CHECKPOINTS.items():
        checkpoint = upstream / "resources" / "anti_spoof_models" / filename
        actual = sha256(checkpoint)
        assert actual == checksum, f"{filename}: sha256 mismatch: {actual}"

        model = model_mapping[model_type](conv6_kernel=kernel)
        state_dict = torch.load(checkpoint, map_location="cpu")
        # strip DataParallel prefix
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()

        output = output_dir / (Path(filename).stem + ".onnx")
        torch.onnx.export(
            model,
            torch.randn(1, 3, 80, 80),
            str(output),
            export_params=True,
            # keep weights embedded: a single self-contained file
            external_data=False,
            opset_version=OPSET,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        validate(model, output)
        print(f"{output.name}: {sha256(output)}")


def validate(model, onnx_file, n_batches=8):
    import onnxruntime
    import torch

    session = onnxruntime.InferenceSession(
        str(onnx_file), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(0)
    for _ in range(n_batches):
        # model was trained on raw 0-255 inputs (no /255 normalization)
        batch = rng.uniform(0, 255, size=(4, 3, 80, 80)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(batch)).numpy()
        actual = session.run(None, {"input": batch})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"{onnx_file.name}: parity OK")


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        upstream = Path(temp) / "upstream"
        clone_upstream(upstream)
        sys.path.insert(0, str(upstream))
        export(upstream, output_dir)


if __name__ == "__main__":
    main()
