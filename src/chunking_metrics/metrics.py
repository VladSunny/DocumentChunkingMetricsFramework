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
    """Not implemented yet
    Block Integrity evaluates how much the boundaries of chunks
    preserve the natural structure of the document.
    The structural blocks can be paragraphs, lists, tables, code blocks, quotations,
    and other elements that can be automatically selected when parsing a document.

    The basic intuition is simple:
    if a table, list, or linked paragraph is split between several chunks,
    part of the local context may be lost during retrieval.

    At the same time, the same gap has different significance for different types of structure:
    cutting a table is usually more dangerous than a long paragraph.
    """
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
    """Contextual Coherence evaluates how consistent
    a chunk is with the local context from which it was extracted.

    Unlike the ICC, this metric does not look inside the chunk,
    but at its relation to the surrounding text of the document.

    A typical scheme is to compare the embedding
    of a chunk with the embedding of a context window around it.

    The context can be adjacent sentences, paragraphs,
    or a fixed number of tokens before and after the chunk.

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


def coreference_integrity(*args: Any, **kwargs: Any) -> None:
    """Not implemented yet
    Coreference Integrity measures how much the boundaries of chunks
    break the relationship between expressions that refer to the same entity.

    For example:

    Chunk 1:
    Ivan handed over the contract to Peter.

    Chunk 2:
    He signed it the next day.

    In the second chunk, the pronouns "he" and "his" require information from the first.
    If only the second chunk is found during retrieval, its interpretation will become ambiguous.

    Coreference resolver builds chains of mentions of a single entity,
    after which you can check which connections pass through the boundaries of chunks.
    """
    del args, kwargs
    raise NotImplementedError("Not implemented yet")


def boundary_clarity(uncond_ppls: np.ndarray[float], cond_ppls: np.ndarray[float]) -> float:
    """Boundary Clarity evaluates how independent two neighboring chunks are
    from the point of view of the causal language model.

    If the previous chunk makes it much easier for the language model to predict the next one,
    this indicates a strong relationship between them and a potentially weak boundary.

    The metric is based on comparing the unconditional perplexity of the next chunk
    and the perplexity of the same text in the presence of the previous chunk in the context.
    """
    uncond_ppls = np.asarray(uncond_ppls)
    cond_ppls = np.asarray(cond_ppls)
    if uncond_ppls.ndim != 1 or cond_ppls.ndim != 1:
        return 0.0
    if cond_ppls.size == 0 or uncond_ppls.size != cond_ppls.size + 1:
        return 0.0
    if not np.all(np.isfinite(uncond_ppls)) or not np.all(np.isfinite(cond_ppls)):
        return 0.0
    if np.any(uncond_ppls <= 0) or np.any(cond_ppls <= 0):
        return 0.0

    return float(np.mean(cond_ppls / uncond_ppls[1:]))


def chunk_score(*args: Any, **kwargs: Any) -> None:
    """ChunkScore is a reference-free composite metric
    that combines two properties of a set of chunks:
    Logical Independence (LI) — to what extent chunks are logically independent from each other;
    Semantic Dispersion (SD) — how semantically diverse a set of chunks is
    and does not consist of almost duplicate fragments.
    General form:
        ChunkScore = lambda * LI + (1 - lambda) * S * D
    """
    del args, kwargs
    raise NotImplementedError("Not implemented yet")


def concept_unity(statements_embs: np.ndarray) -> float:
    """Evaluate the semantic unity of the statements generated from one chunk.

    Args:
        statements_embs: Statement embeddings with shape ``(statements, embedding_dims)``.

    Returns:
        The mean pairwise cosine similarity with negative similarities clipped to zero.
        Invalid or empty arrays produce ``0.0``.
    """
    if statements_embs.ndim != 2 or min(statements_embs.shape) <= 0:
        return 0.0
    if not np.issubdtype(statements_embs.dtype, np.number) or np.iscomplexobj(statements_embs):
        return 0.0
    if not np.all(np.isfinite(statements_embs)):
        return 0.0

    floating_dtype = np.result_type(statements_embs.dtype, np.float64)
    floating_embs = statements_embs.astype(floating_dtype, copy=False)
    scales = np.max(np.abs(floating_embs), axis=1, keepdims=True)
    if np.any(scales == 0):
        return 0.0
    scaled_embs = floating_embs / scales

    similarities = cosine_similarity(
        scaled_embs[:, np.newaxis, :],
        scaled_embs[np.newaxis, :, :],
    )
    if not np.all(np.isfinite(similarities)):
        return 0.0

    return float(np.mean(np.clip(similarities, 0.0, 1.0)))


def semantic_independence(
    standalone_answer_embs: np.ndarray,
    contextual_answer_embs: np.ndarray,
) -> float:
    """Evaluate whether additional context changes answers derived from a chunk.

    Args:
        standalone_answer_embs: Answer embeddings produced from the chunk alone, with shape
            ``(answers, embedding_dims)``.
        contextual_answer_embs: Embeddings of the corresponding answers produced with additional
            context, with the same shape as ``standalone_answer_embs``.

    Returns:
        The mean cosine similarity between corresponding answer embeddings, clipped to
        ``[0.0, 1.0]``. Invalid, empty, or mismatched arrays produce ``0.0``.
    """
    if (
        standalone_answer_embs.ndim != 2
        or contextual_answer_embs.ndim != 2
        or standalone_answer_embs.shape != contextual_answer_embs.shape
        or min(standalone_answer_embs.shape) <= 0
    ):
        return 0.0
    if (
        not np.issubdtype(standalone_answer_embs.dtype, np.number)
        or not np.issubdtype(contextual_answer_embs.dtype, np.number)
        or np.iscomplexobj(standalone_answer_embs)
        or np.iscomplexobj(contextual_answer_embs)
    ):
        return 0.0
    if not np.all(np.isfinite(standalone_answer_embs)) or not np.all(
        np.isfinite(contextual_answer_embs)
    ):
        return 0.0

    floating_dtype = np.result_type(
        standalone_answer_embs.dtype,
        contextual_answer_embs.dtype,
        np.float64,
    )
    standalone_embs = standalone_answer_embs.astype(floating_dtype, copy=False)
    contextual_embs = contextual_answer_embs.astype(floating_dtype, copy=False)
    standalone_scales = np.max(np.abs(standalone_embs), axis=1, keepdims=True)
    contextual_scales = np.max(np.abs(contextual_embs), axis=1, keepdims=True)
    if np.any(standalone_scales == 0) or np.any(contextual_scales == 0):
        return 0.0

    similarities = cosine_similarity(
        standalone_embs / standalone_scales,
        contextual_embs / contextual_scales,
    )
    if not np.all(np.isfinite(similarities)):
        return 0.0

    return float(np.mean(np.clip(similarities, 0.0, 1.0)))
