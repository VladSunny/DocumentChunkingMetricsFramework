import numpy as np


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray | np.float64:
    """Returns cosine similarity along the last axis."""
    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator
