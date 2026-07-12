"""
Handled exceptions raised by FaceEngine
"""


class FaceNotFoundError(Exception):
    """Raised when the face is not found in the image"""


class ModelNotFoundError(Exception):
    """Raised when an explicit model name is not in the registry"""


class TrainError(Exception):
    """Raised when the fit(train) process is failed"""
