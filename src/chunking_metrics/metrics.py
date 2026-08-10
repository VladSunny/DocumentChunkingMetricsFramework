import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def size_compliance(lengths: Iterable[int], min_size: int, max_size: int) -> float:
    """Returns the ratio of the number of chunks that fit the size to the total number of chunks"""
    lengths = np.array(lengths)

    if len(lengths) <= 0:
        return 0
    if min_size < 0:
        return 0
    if max_size < 0:
        return 0
    if max_size < min_size:
        return 0

    min_mask = lengths >= min_size
    max_mask = lengths <= max_size
    relevant_cnt = np.sum(min_mask & max_mask)
    return relevant_cnt / lengths.shape[0]


def block_integrity(*args: Any, **kwargs: Any) -> None:
    """Not implemented yet"""
    del args, kwargs
    raise NotImplementedError("Not implemented yet")


def intrachunk_cohesion(embs: Iterable[np.ndarray]) -> float:
    """Intrachunk Cohesion evaluates the internal semantic uniformity of a chunk.
    A good chunk should contain sentences related to a related topic or a common semantic context.
    - embs -> array (chunks, sentences, emb dims)
    """

    return 0.0
