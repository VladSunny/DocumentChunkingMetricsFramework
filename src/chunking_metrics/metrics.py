import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from .utils import cosine_similarity

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

    embs = list(embs)
    if len(embs) <= 0:
        return 0.0

    chunk_cohesions = []
    for chunk_embs in embs:
        if chunk_embs.ndim != 2 or chunk_embs.shape[0] <= 0:
            return 0.0

        centroid = np.mean(chunk_embs, axis=0)
        similarities = cosine_similarity(chunk_embs, centroid)
        if np.any(np.isnan(similarities)):
            return 0.0

        chunk_cohesions.append(np.mean(similarities))

    return float(np.mean(chunk_cohesions))


def contextual_coherence(chunk_embs: np.ndarray, context_embs: np.ndarray) -> float:
    """Contextual Coherence evaluates how consistent a chunk is with the local context from which it was extracted.
    Unlike the ICC, this metric does not look inside the chunk, but at its relation to the surrounding text of the document.
    A typical scheme is to compare the embedding of a chunk with the embedding of a context window around it.
    The context can be adjacent sentences, paragraphs, or a fixed number of tokens before and after the chunk.
    - chunk_embs -> array (emb dims)
    - context_embs -> array (context blocks, emb dims)
    """
    if chunk_embs.ndim != 1:
        return 0.0
    if context_embs.ndim != 2 or context_embs.shape[0] <= 0:
        return 0.0
    if chunk_embs.shape[0] != context_embs.shape[1]:
        return 0.0

    context_emb = np.mean(context_embs, axis=0)
    similarity = cosine_similarity(chunk_embs, context_emb)
    if np.isnan(similarity):
        return 0.0

    return float(similarity)
