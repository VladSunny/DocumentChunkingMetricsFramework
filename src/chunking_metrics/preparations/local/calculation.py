import warnings
from collections.abc import Sequence

import numpy as np
import torch

from . import utils

DEFAULT_EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_PERPLEXITY_MODEL = "ai-forever/rugpt3small_based_on_gpt2"


def calculate_perplexity(
    text: str,
    model_name: str = DEFAULT_PERPLEXITY_MODEL,
    *,
    hf_token: str | None = None,
    context: str | None = None,
    device: str | None = None,
) -> float:
    """Calculate causal-language-model perplexity for a text chunk.

    Args:
        text: Target text whose tokens contribute to perplexity.
        model_name: Hugging Face model identifier or local model path.
        context: Optional preceding text visible to the model but excluded from the loss.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override. When omitted,
            CUDA is preferred, followed by MPS and CPU.

    Returns:
        The exponentiated mean negative log-likelihood of the target tokens.

    Raises:
        ValueError: If an argument is empty, the model is incompatible, or the target
            does not fit within the model context window.

    The boundary between context and target is normalized to one space. If the combined
    sequence is too long, the oldest context tokens are discarded and a warning is emitted.
    """
    if not text.strip():
        raise ValueError("text must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    resolved_device = utils._resolve_device(device)
    tokenizer, model, max_length = utils._load_model_and_tokenizer(
        model_name.strip(), hf_token, resolved_device
    )

    target_text = f" {text.lstrip()}"
    context_text = context.rstrip() if context is not None else ""
    target_ids = list(tokenizer.encode(target_text, add_special_tokens=False))
    context_ids = (
        list(tokenizer.encode(context_text, add_special_tokens=False)) if context_text else []
    )
    if not target_ids:
        raise ValueError("text does not contain any model tokens")

    prefix_token_id = utils._prefix_token_id(tokenizer, model)
    target_length = 1 + len(target_ids)
    if target_length > max_length:
        raise ValueError(
            f"target requires {target_length} tokens but the model context window is {max_length}"
        )

    available_context_length = max_length - target_length
    discarded_context_tokens = max(0, len(context_ids) - available_context_length)
    if discarded_context_tokens:
        context_ids = context_ids[discarded_context_tokens:]
        warnings.warn(
            f"discarded {discarded_context_tokens} context tokens to fit the model window",
            UserWarning,
            stacklevel=2,
        )

    input_ids = [prefix_token_id, *context_ids, *target_ids]
    labels = [utils._IGNORED_LABEL] * (1 + len(context_ids)) + target_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=resolved_device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=resolved_device)

    with torch.inference_mode():
        output = model(input_ids=input_tensor, labels=label_tensor)
    if output.loss is None:
        raise ValueError("the causal language model did not return a loss")
    return float(torch.exp(output.loss.detach()).cpu().item())


def calculate_embeddings(
    texts: str | Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    hf_token: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Calculate normalized sentence embeddings for one or more texts.

    Args:
        texts: A text or sequence of texts to embed. Texts must contain non-whitespace
            characters.
        model_name: Hugging Face Sentence Transformers model identifier or local model path.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override. When omitted,
            CUDA is preferred, followed by MPS and CPU.
        batch_size: Number of texts encoded in one inference batch.

    Returns:
        A float32 NumPy vector of shape ``(embedding_dim,)`` for one text, or a matrix
        of shape ``(text_count, embedding_dim)`` for a sequence. Every vector is L2-normalized.

    Raises:
        ValueError: If an argument is empty or ``batch_size`` is not positive.

    Texts longer than the model's maximum sequence length are truncated by Sentence
    Transformers after this function emits a warning.
    """
    text_items = [texts] if isinstance(texts, str) else list(texts)
    if not text_items:
        raise ValueError("texts must not be empty")
    for index, text in enumerate(text_items):
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    resolved_device = utils._resolve_device(device)
    model = utils._load_embedding_model(model_name.strip(), resolved_device, hf_token)
    utils._warn_about_embedding_truncation(text_items, model)
    encode_input = texts if isinstance(texts, str) else text_items
    return np.asarray(
        model.encode(
            encode_input,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )


def retrieve_relevant_chunks(
    queries: Sequence[str],
    candidate_chunks: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    top_k: int = 3,
    device: str | None = None,
    batch_size: int = 32,
) -> list[list[str]]:
    """Retrieve the most relevant candidate chunks for each query using embeddings.

    Args:
        queries: Non-empty retrieval queries, such as questions or factual statements.
        candidate_chunks: Non-empty chunks to rank independently for every query.
        model_name: Hugging Face Sentence Transformers model identifier or local model path.
        top_k: Maximum number of chunks returned per query.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override.
        batch_size: Number of texts encoded in one inference batch.

    Returns:
        One list of candidate chunks per query, ordered by decreasing cosine similarity.
        When fewer than ``top_k`` candidates are available, all candidates are returned.
        Equal similarities preserve the original candidate order.

    Raises:
        ValueError: If a sequence or string is empty, or a numeric argument is not positive.

    Query and candidate embeddings are calculated together in one model call. Callers are
    responsible for excluding a primary chunk when HOPE Semantic Independence requires retrieval
    only from the other chunks in the document.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")

    query_items = list(queries)
    candidate_items = list(candidate_chunks)
    if not query_items:
        raise ValueError("queries must not be empty")
    if not candidate_items:
        raise ValueError("candidate_chunks must not be empty")
    for index, query in enumerate(query_items):
        if not query.strip():
            raise ValueError(f"queries[{index}] must not be empty")
    for index, candidate_chunk in enumerate(candidate_items):
        if not candidate_chunk.strip():
            raise ValueError(f"candidate_chunks[{index}] must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_count = len(query_items)
    embeddings = calculate_embeddings(
        [*query_items, *candidate_items],
        model_name,
        device=device,
        batch_size=batch_size,
    )
    similarities = embeddings[:query_count] @ embeddings[query_count:].T
    retrieved_count = min(top_k, len(candidate_items))
    ranked_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :retrieved_count]
    return [[candidate_items[index] for index in query_indices] for query_indices in ranked_indices]
