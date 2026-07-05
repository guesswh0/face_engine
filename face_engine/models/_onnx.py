"""Shared helpers for onnxruntime backed models.

Underscore-prefixed so ``import_package`` skips it: importing it must stay
an explicit choice of modules that already require onnxruntime.
"""

import onnxruntime


def _providers():
    """Cuda when available, cpu otherwise.

    Insightface requests CUDAExecutionProvider unconditionally, making
    onnxruntime warn on every non-CUDA machine.
    """
    available = onnxruntime.get_available_providers()
    return [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ]
