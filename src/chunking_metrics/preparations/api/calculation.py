import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from openai import OpenAI


def calculate_perplexity(
    text: str,
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    *,
    context: str | None = None,
) -> float:
    """Calculate target perplexity through a vLLM OpenAI-compatible Completions API.

    Args:
        text: Target text whose tokens contribute to perplexity.
        model_name: Causal language model identifier served by vLLM.
        api_key: API key passed to the OpenAI-compatible client.
        base_url: Optional OpenAI-compatible API base URL.
        context: Optional preceding text visible to the model but excluded from perplexity.

    Returns:
        The exponentiated mean negative log-probability of the target tokens.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is empty or the API response does not provide consistent,
            finite prompt token log-probabilities and character offsets.

    This helper targets vLLM's ``/v1/completions`` implementation and requires support for
    echoed prompt log-probabilities in ``text_offset`` and ``token_logprobs``. Provider errors,
    including context-window overflow, are propagated without retries or truncation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be a string or None")
    if not text.strip():
        raise ValueError("text must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if base_url is not None and not base_url.strip():
        raise ValueError("base_url must not be empty")
    if context is not None and not context.strip():
        raise ValueError("context must not be empty")

    context_text = context.rstrip() if context is not None else ""
    target_text = f" {text.lstrip()}"
    prompt = f"{context_text}{target_text}"
    target_start = len(context_text)
    target_end = len(prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)
    # A one-token continuation makes vLLM return the echoed prompt logprobs; its score is ignored.
    response = client.completions.create(
        model=model_name.strip(),
        prompt=prompt,
        max_tokens=1,
        echo=True,
        logprobs=0,
        stream=False,
    )

    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("API response must contain a completion choice with prompt logprobs")
    logprobs = getattr(choices[0], "logprobs", None)
    text_offsets = getattr(logprobs, "text_offset", None)
    token_logprobs = getattr(logprobs, "token_logprobs", None)
    if (
        not isinstance(text_offsets, Sequence)
        or isinstance(text_offsets, (str, bytes))
        or not isinstance(token_logprobs, Sequence)
        or isinstance(token_logprobs, (str, bytes))
        or not text_offsets
        or len(text_offsets) != len(token_logprobs)
    ):
        raise ValueError(
            "API response must contain equally sized text_offset and token_logprobs arrays"
        )

    previous_offset = -1
    target_logprobs: list[float] = []
    for index, (offset, token_logprob) in enumerate(zip(text_offsets, token_logprobs, strict=True)):
        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not previous_offset <= offset <= target_end
        ):
            raise ValueError("API response text_offset values must be valid character offsets")
        previous_offset = offset

        if token_logprob is None:
            if index == 0 and offset == 0:
                continue
            raise ValueError("API response token_logprobs may be None only for the first token")
        if (
            not isinstance(token_logprob, Real)
            or isinstance(token_logprob, bool)
            or not math.isfinite(token_logprob)
        ):
            raise ValueError("API response token_logprobs must contain finite numbers")
        if target_start <= offset < target_end:
            target_logprobs.append(float(token_logprob))

    if not target_logprobs:
        raise ValueError("API response does not contain target token logprobs")
    return math.exp(-sum(target_logprobs) / len(target_logprobs))


def calculate_embeddings(
    texts: str | Sequence[str],
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    *,
    dimensions: int | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """Calculate embeddings through an OpenAI-compatible Embeddings endpoint.

    Args:
        texts: A text or non-empty sequence of texts to embed.
        model_name: Embedding model identifier understood by the API provider.
        api_key: API key passed to the OpenAI-compatible client.
        base_url: Optional OpenAI-compatible API base URL.
        dimensions: Optional positive output embedding dimension.
        batch_size: Number of texts sent in each request, from 1 through 2048.

    Returns:
        An unnormalized float32 vector of shape ``(embedding_dim,)`` for one string, or a
        matrix of shape ``(text_count, embedding_dim)`` for a sequence.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument or API response violates the required contract.

    One client is reused for sequential batches. Provider errors are not caught, requests are
    not retried, and no partial result is returned.
    """
    if not isinstance(texts, (str, Sequence)):
        raise TypeError("texts must be a string or a sequence of strings")
    text_items = [texts] if isinstance(texts, str) else list(texts)
    if not text_items:
        raise ValueError("texts must not be empty")
    for index, text in enumerate(text_items):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a string")
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be empty")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    if dimensions is not None and (not isinstance(dimensions, int) or isinstance(dimensions, bool)):
        raise TypeError("dimensions must be an integer or None")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if base_url is not None and not base_url.strip():
        raise ValueError("base_url must not be empty")
    if dimensions is not None and dimensions <= 0:
        raise ValueError("dimensions must be greater than zero")
    if not 1 <= batch_size <= 2048:
        raise ValueError("batch_size must be between 1 and 2048")

    client = OpenAI(api_key=api_key, base_url=base_url)
    embedding_batches: list[np.ndarray] = []
    embedding_dimension: int | None = None
    for start in range(0, len(text_items), batch_size):
        batch = text_items[start : start + batch_size]
        request: dict[str, object] = {
            "model": model_name.strip(),
            "input": batch,
            "encoding_format": "float",
        }
        if dimensions is not None:
            request["dimensions"] = dimensions
        response = client.embeddings.create(**request)
        data = getattr(response, "data", None)
        if not isinstance(data, Sequence) or len(data) != len(batch):
            raise ValueError("API response must contain one embedding per input")

        ordered_embeddings: list[np.ndarray | None] = [None] * len(batch)
        indices: list[int] = []
        for item in data:
            index = getattr(item, "index", None)
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(
                    f"API response must contain embedding indices from 0 to {len(batch) - 1}"
                )
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("API response must contain a unique embedding index for each input")
        if set(indices) != set(range(len(batch))):
            raise ValueError(
                f"API response must contain embedding indices from 0 to {len(batch) - 1}"
            )

        batch_dimension: int | None = None
        for item, index in zip(data, indices, strict=True):
            embedding = getattr(item, "embedding", None)
            if not isinstance(embedding, Sequence) or any(
                not isinstance(value, Real) or isinstance(value, bool) for value in embedding
            ):
                raise ValueError("API response embeddings must contain finite numbers")
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError("API response embeddings must have the same non-zero dimension")
            if batch_dimension is None:
                batch_dimension = vector.size
            elif vector.size != batch_dimension:
                raise ValueError("API response embeddings must have the same non-zero dimension")
            if not np.isfinite(vector).all():
                raise ValueError("API response embeddings must contain finite numbers")
            ordered_embeddings[index] = vector

        if embedding_dimension is None:
            embedding_dimension = batch_dimension
        elif batch_dimension != embedding_dimension:
            raise ValueError("API response embeddings must have the same dimension across batches")
        embedding_batches.append(np.stack(ordered_embeddings))

    result = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    return result[0] if isinstance(texts, str) else result
