import json
import random
import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import utils
from .prompts import (
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_ANSWER_SYSTEM_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_QUESTION_SYSTEM_PROMPT,
    DEFAULT_STATEMENT_PROMPT,
    DEFAULT_STATEMENT_SYSTEM_PROMPT,
)

DEFAULT_EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_PERPLEXITY_MODEL = "ai-forever/rugpt3small_based_on_gpt2"
DEFAULT_STATEMENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_IGNORED_LABEL = -100
_UNBOUNDED_MODEL_LENGTH = 1_000_000


def _resolve_device(device: str | None) -> str:
    if device is None:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid device: {device!r}") from error

    if resolved_device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return str(resolved_device)


def _model_max_length(tokenizer: Any, model: Any) -> int:
    candidates = (
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    )
    finite_lengths = [
        length
        for length in candidates
        if isinstance(length, int) and 0 < length < _UNBOUNDED_MODEL_LENGTH
    ]
    if not finite_lengths:
        raise ValueError("could not determine the model context window")
    return min(finite_lengths)


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(model_name: str, device: str) -> tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model, _model_max_length(tokenizer, model)


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    return model


def _warn_about_embedding_truncation(texts: list[str], model: Any) -> None:
    max_length = model.max_seq_length
    if not isinstance(max_length, int) or max_length <= 0:
        return

    tokenized = model.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    truncated_count = sum(len(input_ids) > max_length for input_ids in tokenized["input_ids"])
    if truncated_count:
        noun = "text exceeds" if truncated_count == 1 else "texts exceed"
        warnings.warn(
            f"{truncated_count} {noun} the model limit of {max_length} tokens "
            "and will be truncated",
            UserWarning,
            stacklevel=2,
        )


def _statement_messages(
    chunk: str,
    statement_count: int,
    prompt: str,
) -> list[dict[str, str]]:
    try:
        user_prompt = prompt.format(
            chunk=json.dumps(chunk, ensure_ascii=False),
            statement_count=statement_count,
        )
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {
            "role": "system",
            "content": DEFAULT_STATEMENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _parse_statement_response(response: str, statement_count: int) -> list[str]:
    # print("!!!!", response)
    error_message = (
        f"model response must be a JSON array of exactly {statement_count} non-empty strings"
    )
    try:
        statements = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(error_message) from error
    if (
        not isinstance(statements, list)
        or len(statements) != statement_count
        or any(not isinstance(statement, str) or not statement.strip() for statement in statements)
    ):
        raise ValueError(error_message)
    return [statement.strip() for statement in statements]


def _information_preservation_messages(
    segment: str,
    prompt: str,
) -> list[dict[str, str]]:
    try:
        user_prompt = prompt.format(segment=json.dumps(segment, ensure_ascii=False))
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {
            "role": "system",
            "content": DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_prompt},
    ]


def _parse_information_preservation_response(response: object) -> tuple[str, list[str]]:
    error_message = (
        "model response must be a JSON object with one non-empty true_statement and exactly "
        "three distinct non-empty false_statements"
    )
    if not isinstance(response, str):
        raise ValueError(error_message)
    try:
        statements = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(error_message) from error
    if not isinstance(statements, dict) or set(statements) != {
        "true_statement",
        "false_statements",
    }:
        raise ValueError(error_message)

    true_statement = statements["true_statement"]
    false_statements = statements["false_statements"]
    if (
        not isinstance(true_statement, str)
        or not true_statement.strip()
        or not isinstance(false_statements, list)
        or len(false_statements) != 3
        or any(
            not isinstance(statement, str) or not statement.strip()
            for statement in false_statements
        )
    ):
        raise ValueError(error_message)

    true_statement = true_statement.strip()
    false_statements = [statement.strip() for statement in false_statements]
    if len(set(false_statements)) != 3 or true_statement in false_statements:
        raise ValueError(error_message)
    return true_statement, false_statements


def _information_preservation_evaluation_messages(
    statements: Sequence[str],
    relevant_chunks: Sequence[str],
    prompt: str,
) -> list[dict[str, str]]:
    numbered_statements = [
        {"index": index, "statement": statement}
        for index, statement in enumerate(statements, start=1)
    ]
    try:
        user_prompt = prompt.format(
            statements=json.dumps(numbered_statements, ensure_ascii=False),
            relevant_chunks=json.dumps(relevant_chunks, ensure_ascii=False),
        )
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {
            "role": "system",
            "content": DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_prompt},
    ]


def _parse_information_preservation_evaluation_response(response: object) -> int:
    error_message = (
        "model response must be a JSON object containing only selected_index from 1 to 4"
    )
    if not isinstance(response, str):
        raise ValueError(error_message)
    try:
        selection = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(error_message) from error
    if not isinstance(selection, dict) or set(selection) != {"selected_index"}:
        raise ValueError(error_message)
    selected_index = selection["selected_index"]
    if (
        not isinstance(selected_index, int)
        or isinstance(selected_index, bool)
        or not 1 <= selected_index <= 4
    ):
        raise ValueError(error_message)
    return selected_index


def _question_messages(
    chunk: str,
    question_count: int,
    prompt: str,
) -> list[dict[str, str]]:
    try:
        user_prompt = prompt.format(
            chunk=json.dumps(chunk, ensure_ascii=False),
            question_count=question_count,
        )
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {
            "role": "system",
            "content": DEFAULT_QUESTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _parse_question_response(response: str, question_count: int) -> list[str]:
    error_message = (
        f"model response must be a JSON array of exactly {question_count} non-empty strings"
    )
    try:
        questions = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(error_message) from error
    if (
        not isinstance(questions, list)
        or len(questions) != question_count
        or any(not isinstance(question, str) or not question.strip() for question in questions)
    ):
        raise ValueError(error_message)
    return [question.strip() for question in questions]


def _answer_messages(
    question: str,
    chunk: str,
    additional_chunks: Sequence[str],
    prompt: str,
) -> list[dict[str, str]]:
    try:
        user_prompt = prompt.format(
            question=json.dumps(question, ensure_ascii=False),
            chunk=json.dumps(chunk, ensure_ascii=False),
            additional_chunks=json.dumps(list(additional_chunks), ensure_ascii=False),
        )
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {"role": "system", "content": DEFAULT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _clean_answer(response: object, question_index: int) -> str:
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"answer for question {question_index} must not be empty")
    return response.strip()


def _prefix_token_id(tokenizer: Any, model: Any) -> int:
    candidates = (
        getattr(tokenizer, "bos_token_id", None),
        getattr(model.config, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.config, "eos_token_id", None),
    )
    for token_id in candidates:
        if isinstance(token_id, int):
            return token_id
    raise ValueError("the model must define a BOS or EOS token")


def calculate_perplexity(
    text: str,
    model_name: str = DEFAULT_PERPLEXITY_MODEL,
    *,
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
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is empty, the model is incompatible, or the target
            does not fit within the model context window.

    The boundary between context and target is normalized to one space. If the combined
    sequence is too long, the oldest context tokens are discarded and a warning is emitted.
    """
    utils._validate_arguments(text, model_name, context, device)
    resolved_device = _resolve_device(device)
    tokenizer, model, max_length = _load_model_and_tokenizer(model_name.strip(), resolved_device)

    target_text = f" {text.lstrip()}"
    context_text = context.rstrip() if context is not None else ""
    target_ids = list(tokenizer.encode(target_text, add_special_tokens=False))
    context_ids = (
        list(tokenizer.encode(context_text, add_special_tokens=False)) if context_text else []
    )
    if not target_ids:
        raise ValueError("text does not contain any model tokens")

    prefix_token_id = _prefix_token_id(tokenizer, model)
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
    labels = [_IGNORED_LABEL] * (1 + len(context_ids)) + target_ids
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
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is empty or ``batch_size`` is not positive.

    Texts longer than the model's maximum sequence length are truncated by Sentence
    Transformers after this function emits a warning.
    """
    text_items = utils._validate_embedding_arguments(texts, model_name, device, batch_size)
    resolved_device = _resolve_device(device)
    model = _load_embedding_model(model_name.strip(), resolved_device)
    _warn_about_embedding_truncation(text_items, model)
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
        TypeError: If an argument has an invalid type.
        ValueError: If a sequence or string is empty, or a numeric argument is not positive.

    Query and candidate embeddings are calculated together in one model call. Callers are
    responsible for excluding a primary chunk when HOPE Semantic Independence requires retrieval
    only from the other chunks in the document.
    """
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise TypeError("queries must be a sequence of strings")
    if isinstance(candidate_chunks, (str, bytes)) or not isinstance(candidate_chunks, Sequence):
        raise TypeError("candidate_chunks must be a sequence of strings")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")

    query_items = list(queries)
    candidate_items = list(candidate_chunks)
    if not query_items:
        raise ValueError("queries must not be empty")
    if not candidate_items:
        raise ValueError("candidate_chunks must not be empty")
    for index, query in enumerate(query_items):
        if not isinstance(query, str):
            raise TypeError(f"queries[{index}] must be a string")
        if not query.strip():
            raise ValueError(f"queries[{index}] must not be empty")
    for index, candidate_chunk in enumerate(candidate_items):
        if not isinstance(candidate_chunk, str):
            raise TypeError(f"candidate_chunks[{index}] must be a string")
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


def generate_answers_local(
    questions: Sequence[str],
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    additional_chunks_by_question: Sequence[Sequence[str]] | None = None,
    prompt: str = DEFAULT_ANSWER_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 128,
    device: str | None = None,
) -> list[str]:
    """Answer questions independently with a local causal chat model.

    Args:
        questions: Non-empty questions to answer, in output order.
        chunk: Required primary source used for every question.
        model_name: Hugging Face causal language model identifier or local model path. Its
            tokenizer must define a chat template.
        additional_chunks_by_question: Optional per-question sequences of extra sources. The
            outer sequence must have the same length as ``questions``.
        prompt: User-message format string containing ``{question}``, ``{chunk}``, and
            ``{additional_chunks}`` placeholders. Values are inserted as JSON.
        temperature: Non-negative generation temperature. Zero selects greedy decoding.
        max_new_tokens: Maximum number of tokens generated for each answer.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override.

    Returns:
        One stripped, non-empty answer per question, preserving question order.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, a prompt and response budget exceed the context
            window, the tokenizer has no chat template, or an answer is empty.

    The model is loaded once, but each question is generated independently without truncation or
    retries.
    """
    question_items, additional_chunk_items = utils._validate_answer_arguments(
        questions,
        chunk,
        model_name,
        additional_chunks_by_question,
        prompt,
        temperature,
        max_new_tokens,
        device,
    )
    resolved_device = _resolve_device(device)
    tokenizer, model, max_length = _load_model_and_tokenizer(model_name.strip(), resolved_device)
    answers: list[str] = []
    for index, (question, additional_chunks) in enumerate(
        zip(question_items, additional_chunk_items, strict=True)
    ):
        messages = _answer_messages(question, chunk, additional_chunks, prompt)
        try:
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except ValueError as error:
            raise ValueError(
                f"question {index}: model tokenizer must define a chat template"
            ) from error
        model_inputs = {name: tensor.to(resolved_device) for name, tensor in model_inputs.items()}
        prompt_length = model_inputs["input_ids"].shape[-1]
        if prompt_length + max_new_tokens > max_length:
            raise ValueError(
                f"question {index} and generated answer do not fit within the model context window"
            )

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_arguments: dict[str, Any] = {
            **model_inputs,
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if temperature > 0:
            generation_arguments["temperature"] = temperature
        with torch.inference_mode():
            output_ids = model.generate(**generation_arguments)
        response = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
        answers.append(_clean_answer(response, index))
    return answers


def generate_answers_api(
    questions: Sequence[str],
    chunk: str,
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    *,
    additional_chunks_by_question: Sequence[Sequence[str]] | None = None,
    prompt: str = DEFAULT_ANSWER_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 128,
) -> list[str]:
    """Answer questions independently through OpenAI-compatible Chat Completions.

    Args:
        questions: Non-empty questions to answer, in output order.
        chunk: Required primary source used for every question.
        model_name: Model identifier understood by the API provider.
        api_key: API key passed to the OpenAI-compatible client.
        base_url: Optional OpenAI-compatible API base URL.
        additional_chunks_by_question: Optional per-question sequences of extra sources. The
            outer sequence must have the same length as ``questions``.
        prompt: User-message format string containing ``{question}``, ``{chunk}``, and
            ``{additional_chunks}`` placeholders. Values are inserted as JSON.
        temperature: Non-negative generation temperature.
        max_new_tokens: Maximum number of tokens generated for each answer.

    Returns:
        One stripped, non-empty answer per question, preserving question order.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid or an API response has empty message content.

    A single client is reused for sequential, independent calls. The function does not retry and
    does not return partial results.
    """
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    question_items, additional_chunk_items = utils._validate_answer_arguments(
        questions,
        chunk,
        model_name,
        additional_chunks_by_question,
        prompt,
        temperature,
        max_new_tokens,
        None,
    )
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if base_url is not None and not base_url.strip():
        raise ValueError("base_url must not be empty")

    client = OpenAI(api_key=api_key, base_url=base_url)
    answers: list[str] = []
    for index, (question, additional_chunks) in enumerate(
        zip(question_items, additional_chunk_items, strict=True)
    ):
        response = client.chat.completions.create(
            model=model_name.strip(),
            messages=_answer_messages(question, chunk, additional_chunks, prompt),
            stream=False,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        answers.append(_clean_answer(response.choices[0].message.content, index))
    return answers


def generate_statements_local(
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_STATEMENT_PROMPT,
    statement_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str | None = None,
) -> list[str]:
    """Generate factual statements from one text chunk with a causal chat model.

    Args:
        chunk: Source text from which the statements are extracted.
        model_name: Hugging Face causal language model identifier or local model path. Its
            tokenizer must define a chat template.
        prompt: Format string used for the user message. It must contain ``{chunk}`` and
            ``{statement_count}`` placeholders. Escape literal braces by doubling them.
        statement_count: Exact number of statements required in the model response.
        temperature: Positive non-zero sampling temperature used to encourage concept coverage.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override. When omitted,
            CUDA is preferred, followed by MPS and CPU.

    Returns:
        A list containing exactly ``statement_count`` non-empty statements.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic and happens once without retries. The chunk is never truncated.
    """
    utils._validate_statement_arguments(
        chunk,
        model_name,
        prompt,
        statement_count,
        temperature,
        max_new_tokens,
        device,
    )
    messages = _statement_messages(chunk, statement_count, prompt)
    resolved_device = _resolve_device(device)
    tokenizer, model, max_length = _load_model_and_tokenizer(model_name.strip(), resolved_device)
    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except ValueError as error:
        raise ValueError("model tokenizer must define a chat template") from error
    model_inputs = {name: tensor.to(resolved_device) for name, tensor in model_inputs.items()}
    prompt_length = model_inputs["input_ids"].shape[-1]
    if prompt_length + max_new_tokens > max_length:
        raise ValueError("chunk and generated response do not fit within the model context window")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
    response = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    return _parse_statement_response(response, statement_count)


def generate_statements_api(
    chunk: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    *,
    prompt: str = DEFAULT_STATEMENT_PROMPT,
    statement_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate factual statements from one text chunk with a causal chat model.

    Args:
        chunk: Source text from which the statements are extracted.
        model_name
        api_key
        prompt: Format string used for the user message. It must contain ``{chunk}`` and
            ``{statement_count}`` placeholders. Escape literal braces by doubling them.
        statement_count: Exact number of statements required in the model response.
        temperature: Positive non-zero sampling temperature used to encourage concept coverage.
        max_new_tokens: Maximum number of tokens available for the JSON response.

    Returns:
        A list containing exactly ``statement_count`` non-empty statements.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic and happens once without retries. The chunk is never truncated.
    """
    messages = _statement_messages(chunk, statement_count, prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        # reasoning_effort="low",
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content

    return _parse_statement_response(content, statement_count)


def generate_information_preservation_statements_api(
    segment: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str | None = None,
    *,
    prompt: str = DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> tuple[str, list[str]]:
    """Generate one true and three false statements for HOPE Information Preservation.

    Args:
        segment: Source document segment from which the statements are generated.
        model_name: OpenAI-compatible chat model identifier.
        api_key: API key passed to the OpenAI-compatible client.
        base_url: Optional OpenAI-compatible API base URL.
        prompt: User-message format string containing the ``{segment}`` placeholder.
        temperature: Positive non-zero sampling temperature.
        max_new_tokens: Maximum number of tokens available for the JSON response.

    Returns:
        The non-empty true statement and a list of exactly three distinct non-empty false
        statements.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument or the model response violates the required contract.
    """
    utils._validate_information_preservation_arguments(
        segment,
        model_name,
        api_key,
        base_url,
        prompt,
        temperature,
        max_new_tokens,
    )
    messages = _information_preservation_messages(segment, prompt)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return _parse_information_preservation_response(content)


def evaluate_information_preservation_api(
    true_statement: str,
    false_statements: Sequence[str],
    relevant_chunks: Sequence[str],
    model_name: str,
    api_key: str,
    base_url: str | None = None,
    *,
    prompt: str = DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    seed: int | None = None,
) -> int:
    """Run one HOPE Information Preservation multiple-choice evaluation.

    Args:
        true_statement: One non-empty statement known to be true.
        false_statements: Exactly three distinct non-empty false statements. None may equal the
            true statement.
        relevant_chunks: Non-empty chunks retrieved as evidence for the true statement.
        model_name: OpenAI-compatible chat model identifier.
        api_key: API key passed to the OpenAI-compatible client.
        base_url: Optional OpenAI-compatible API base URL.
        prompt: User-message format string containing ``{statements}`` and
            ``{relevant_chunks}``. Both values are inserted as JSON.
        temperature: Non-negative generation temperature.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        seed: Optional seed used only to shuffle the four statements reproducibly.

    Returns:
        ``1`` when the model selects the true statement, otherwise ``0``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument or the model response violates the required contract.

    The function performs exactly one API request without retries. Callers are responsible for
    averaging the results of multiple independently generated tests.
    """
    true_statement_item, false_statement_items, relevant_chunk_items = (
        utils._validate_information_preservation_evaluation_arguments(
            true_statement,
            false_statements,
            relevant_chunks,
            model_name,
            api_key,
            base_url,
            prompt,
            temperature,
            max_new_tokens,
            seed,
        )
    )
    statements = [
        (true_statement_item, True),
        *((statement, False) for statement in false_statement_items),
    ]
    random.Random(seed).shuffle(statements)
    true_statement_index = next(
        index for index, (_, is_true) in enumerate(statements, start=1) if is_true
    )
    messages = _information_preservation_evaluation_messages(
        [statement for statement, _ in statements],
        relevant_chunk_items,
        prompt,
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    print(response.choices[0].message.content)
    selected_index = _parse_information_preservation_evaluation_response(
        response.choices[0].message.content
    )
    return int(selected_index == true_statement_index)


def generate_questions_local(
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_QUESTION_PROMPT,
    question_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str | None = None,
) -> list[str]:
    """Generate questions answerable from one text chunk with a causal chat model.

    Args:
        chunk: Source text from which the questions are generated.
        model_name: Hugging Face causal language model identifier or local model path. Its
            tokenizer must define a chat template.
        prompt: Format string used for the user message. It must contain ``{chunk}`` and
            ``{question_count}`` placeholders. Escape literal braces by doubling them.
        question_count: Exact number of questions required in the model response.
        temperature: Positive non-zero sampling temperature used to encourage question diversity.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override. When omitted,
            CUDA is preferred, followed by MPS and CPU.

    Returns:
        A list containing exactly ``question_count`` non-empty questions.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic and happens once without retries. The chunk is never truncated.
    """
    utils._validate_question_arguments(
        chunk,
        model_name,
        prompt,
        question_count,
        temperature,
        max_new_tokens,
        device,
    )
    messages = _question_messages(chunk, question_count, prompt)
    resolved_device = _resolve_device(device)
    tokenizer, model, max_length = _load_model_and_tokenizer(model_name.strip(), resolved_device)
    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except ValueError as error:
        raise ValueError("model tokenizer must define a chat template") from error
    model_inputs = {name: tensor.to(resolved_device) for name, tensor in model_inputs.items()}
    prompt_length = model_inputs["input_ids"].shape[-1]
    if prompt_length + max_new_tokens > max_length:
        raise ValueError("chunk and generated response do not fit within the model context window")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
    response = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
    print(response)
    return _parse_question_response(response, question_count)


def generate_questions_api(
    chunk: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    *,
    prompt: str = DEFAULT_QUESTION_PROMPT,
    question_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate questions answerable from one text chunk with a causal chat model.

    Args:
        chunk: Source text from which the questions are generated.
        model_name: Hugging Face causal language model identifier or local model path. Its
            tokenizer must define a chat template.
        prompt: Format string used for the user message. It must contain ``{chunk}`` and
            ``{question_count}`` placeholders. Escape literal braces by doubling them.
        question_count: Exact number of questions required in the model response.
        temperature: Positive non-zero sampling temperature used to encourage question diversity.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override. When omitted,
            CUDA is preferred, followed by MPS and CPU.

    Returns:
        A list containing exactly ``question_count`` non-empty questions.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic and happens once without retries. The chunk is never truncated.
    """

    messages = _question_messages(chunk, question_count, prompt)

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        # reasoning_effort="low",
        extra_body={"thinking": {"type": "disabled"}},
        response_format={"type": "json_object"},
        max_tokens=max_new_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content

    return _parse_question_response(content, question_count)
