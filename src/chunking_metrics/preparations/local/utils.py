import json
import math
import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeVar

import torch
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

_IGNORED_LABEL = -100
_UNBOUNDED_MODEL_LENGTH = 1_000_000
_Response = TypeVar("_Response", bound=BaseModel)


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
def _load_model_and_tokenizer(model_name: str, hf_token: str | None, device: str) -> tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model = model.to(device)
    model.eval()
    return tokenizer, model, _model_max_length(tokenizer, model)


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str, device: str, hf_token: str | None) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True, token=hf_token)
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


def _generate_text(
    messages: list[dict[str, str]],
    model_name: str,
    *,
    hf_token: str | None = None,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
    chat_template_error: str,
    context_window_error: str,
) -> str:
    resolved_device = _resolve_device(device)
    tokenizer, model, max_length = _load_model_and_tokenizer(model_name.strip(), hf_token, resolved_device)
    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except ValueError as error:
        raise ValueError(chat_template_error) from error
    model_inputs = {name: tensor.to(resolved_device) for name, tensor in model_inputs.items()}
    prompt_length = model_inputs["input_ids"].shape[-1]
    if prompt_length + max_new_tokens > max_length:
        raise ValueError(context_window_error)

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
    return tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)


def _validate_max_regenerations(max_regenerations: int) -> None:
    if not isinstance(max_regenerations, int) or isinstance(max_regenerations, bool):
        raise TypeError("max_regenerations must be an integer")
    if max_regenerations < 0:
        raise ValueError("max_regenerations must be greater than or equal to zero")


def _validate_generation_limits(
    temperature: float,
    max_new_tokens: int,
    *,
    allow_zero_temperature: bool,
) -> None:
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    minimum_is_valid = temperature >= 0 if allow_zero_temperature else temperature > 0
    if not minimum_is_valid:
        qualifier = (
            "greater than or equal to zero" if allow_zero_temperature else "greater than zero"
        )
        raise ValueError(f"temperature must be {qualifier}")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")


def _require_non_empty(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _regeneration_messages(
    original_messages: list[dict[str, str]],
    invalid_response: object,
    contract: str,
) -> list[dict[str, str]]:
    response_text = (
        invalid_response
        if isinstance(invalid_response, str)
        else json.dumps(invalid_response, ensure_ascii=False)
    )
    return [
        *original_messages,
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": f"The previous response was invalid. Return {contract}."},
    ]


def _generate_validated(
    original_messages: list[dict[str, str]],
    response_schema: type[_Response],
    validation_context: dict[str, object] | None,
    *,
    model_name: str,
    hf_token: str | None,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
    max_regenerations: int,
    contract: str,
    error_message: str,
    chat_template_error: str,
    context_window_error: str,
    json_response: bool,
) -> _Response:
    messages = original_messages
    for attempt in range(max_regenerations + 1):
        response = _generate_text(
            messages,
            model_name,
            hf_token=hf_token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            chat_template_error=chat_template_error,
            context_window_error=context_window_error,
        )
        try:
            if json_response:
                return response_schema.model_validate_json(response, context=validation_context)
            return response_schema.model_validate(response, context=validation_context)
        except ValidationError as error:
            if attempt == max_regenerations:
                raise ValueError(error_message) from error
            messages = _regeneration_messages(original_messages, response, contract)
    raise RuntimeError("unreachable")


def _validate_answer_inputs(
    questions: Sequence[str],
    chunk: str,
    model_name: str,
    additional_chunks_by_question: Sequence[Sequence[str]] | None,
    temperature: float,
    max_new_tokens: int,
) -> tuple[list[str], list[list[str]]]:
    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be a sequence of strings")
    question_items = list(questions)
    if not question_items:
        raise ValueError("questions must not be empty")
    for index, question in enumerate(question_items):
        _require_non_empty(question, f"questions[{index}]")
    _require_non_empty(chunk, "chunk")
    _require_non_empty(model_name, "model_name")
    _validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=True)
    if additional_chunks_by_question is None:
        return question_items, [[] for _ in question_items]
    if len(additional_chunks_by_question) != len(question_items):
        raise ValueError("additional_chunks_by_question must contain one item per question")
    additional_chunk_items: list[list[str]] = []
    for question_index, additional_chunks in enumerate(additional_chunks_by_question):
        if isinstance(additional_chunks, (str, bytes)) or not isinstance(
            additional_chunks, Sequence
        ):
            raise TypeError(
                f"additional_chunks_by_question[{question_index}] must be a sequence of strings"
            )
        chunk_items = list(additional_chunks)
        for chunk_index, additional_chunk in enumerate(chunk_items):
            _require_non_empty(
                additional_chunk,
                f"additional_chunks_by_question[{question_index}][{chunk_index}]",
            )
        additional_chunk_items.append(chunk_items)
    return question_items, additional_chunk_items


def _validate_list_generation_inputs(
    chunk: str,
    model_name: str,
    count: int,
    count_name: str,
    temperature: float,
    max_new_tokens: int,
) -> None:
    _require_non_empty(chunk, "chunk")
    _require_non_empty(model_name, "model_name")
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"{count_name} must be an integer")
    if count <= 0:
        raise ValueError(f"{count_name} must be greater than zero")
    _validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=False)


def _validate_evaluation_inputs(
    true_statement: str,
    false_statements: Sequence[str],
    relevant_chunks: Sequence[str],
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str, list[str], list[str]]:
    true_statement_item = _require_non_empty(true_statement, "true_statement")
    false_statement_items = list(false_statements)
    if len(false_statement_items) != 3:
        raise ValueError("false_statements must contain exactly three statements")
    for index, statement in enumerate(false_statement_items):
        if not statement.strip():
            raise ValueError(f"false_statements[{index}] must not be empty")
    false_statement_items = [statement.strip() for statement in false_statement_items]
    if len(set(false_statement_items)) != 3:
        raise ValueError("false_statements must be distinct")
    if true_statement_item in false_statement_items:
        raise ValueError("false_statements must not contain true_statement")
    relevant_chunk_items = list(relevant_chunks)
    if not relevant_chunk_items:
        raise ValueError("relevant_chunks must not be empty")
    for index, chunk in enumerate(relevant_chunk_items):
        if not chunk.strip():
            raise ValueError(f"relevant_chunks[{index}] must not be empty")
    relevant_chunk_items = [chunk.strip() for chunk in relevant_chunk_items]
    _require_non_empty(model_name, "model_name")
    _validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=True)
    return true_statement_item, false_statement_items, relevant_chunk_items
