import json
import math
import warnings
from collections.abc import Sequence
from functools import lru_cache
from string import Formatter
from typing import Any, overload

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import (
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_QUESTION_SYSTEM_PROMPT,
    DEFAULT_STATEMENT_PROMPT,
    DEFAULT_STATEMENT_SYSTEM_PROMPT,
)

DEFAULT_EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
DEFAULT_PERPLEXITY_MODEL = "ai-forever/rugpt3small_based_on_gpt2"
DEFAULT_STATEMENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# TODO: generating using LLM API
# OpenAI api

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


def _validate_embedding_arguments(
    texts: str | Sequence[str],
    model_name: str,
    device: str | None,
    batch_size: int,
) -> list[str]:
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
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string or None")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    return text_items


def _validate_statement_arguments(
    chunk: str,
    model_name: str,
    prompt: str,
    statement_count: int,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
) -> None:
    if not isinstance(chunk, str):
        raise TypeError("chunk must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(statement_count, int) or isinstance(statement_count, bool):
        raise TypeError("statement_count must be an integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string or None")
    if not chunk.strip():
        raise ValueError("chunk must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    try:
        prompt_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(prompt)
            if field_name is not None
        }
    except ValueError as error:
        raise ValueError("prompt must be a valid format string") from error
    required_fields = {"chunk", "statement_count"}
    if not required_fields.issubset(prompt_fields):
        raise ValueError("prompt must contain {chunk} and {statement_count}")
    unsupported_fields = prompt_fields - required_fields
    if unsupported_fields:
        unsupported_field = sorted(unsupported_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")
    if statement_count <= 0:
        raise ValueError("statement_count must be greater than zero")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")


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
    print("!!!!", response)
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


def _validate_question_arguments(
    chunk: str,
    model_name: str,
    prompt: str,
    question_count: int,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
) -> None:
    if not isinstance(chunk, str):
        raise TypeError("chunk must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(question_count, int) or isinstance(question_count, bool):
        raise TypeError("question_count must be an integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string or None")
    if not chunk.strip():
        raise ValueError("chunk must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    try:
        prompt_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(prompt)
            if field_name is not None
        }
    except ValueError as error:
        raise ValueError("prompt must be a valid format string") from error
    required_fields = {"chunk", "question_count"}
    if not required_fields.issubset(prompt_fields):
        raise ValueError("prompt must contain {chunk} and {question_count}")
    unsupported_fields = prompt_fields - required_fields
    if unsupported_fields:
        unsupported_field = sorted(unsupported_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")
    if question_count <= 0:
        raise ValueError("question_count must be greater than zero")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")


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


def _validate_arguments(
    text: str,
    model_name: str,
    context: str | None,
    device: str | None,
) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if context is not None and not isinstance(context, str):
        raise TypeError("context must be a string or None")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string or None")
    if not text.strip():
        raise ValueError("text must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")


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
    _validate_arguments(text, model_name, context, device)
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


@overload
def calculate_embeddings(
    texts: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray: ...


@overload
def calculate_embeddings(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    device: str | None = None,
    batch_size: int = 32,
) -> np.ndarray: ...


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
    text_items = _validate_embedding_arguments(texts, model_name, device, batch_size)
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


def generate_statements(
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
    _validate_statement_arguments(
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


def generate_questions(
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
    _validate_question_arguments(
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
