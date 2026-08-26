import json
import math
import random
from collections.abc import Sequence
from typing import Any, TypeVar

import torch
from pydantic import BaseModel, ValidationError

from ...prompts import (
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
from .._generation import (
    _build_messages,
    _InformationPreservationEvaluationResponse,
    _InformationPreservationResponse,
    _NonEmptyTextResponse,
    _StringListResponse,
)
from .calculation import _load_model_and_tokenizer, _resolve_device

DEFAULT_STATEMENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_Response = TypeVar("_Response", bound=BaseModel)


def _generate_text(
    messages: list[dict[str, str]],
    model_name: str,
    *,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
    chat_template_error: str,
    context_window_error: str,
) -> str:
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


def generate_answers(
    questions: Sequence[str],
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    additional_chunks_by_question: Sequence[Sequence[str]] | None = None,
    prompt: str = DEFAULT_ANSWER_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 128,
    device: str | None = None,
    max_regenerations: int = 0,
) -> list[str]:
    """Answer questions independently with a local causal chat model."""
    _validate_max_regenerations(max_regenerations)
    question_items, additional_chunk_items = _validate_answer_inputs(
        questions,
        chunk,
        model_name,
        additional_chunks_by_question,
        temperature,
        max_new_tokens,
    )
    answers: list[str] = []
    for index, (question, additional_chunks) in enumerate(
        zip(question_items, additional_chunk_items, strict=True)
    ):
        messages = _build_messages(
            DEFAULT_ANSWER_SYSTEM_PROMPT,
            prompt,
            {
                "question": question,
                "chunk": chunk,
                "additional_chunks": additional_chunks,
            },
        )
        response = _generate_validated(
            messages,
            _NonEmptyTextResponse,
            None,
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            max_regenerations=max_regenerations,
            contract="a non-empty text answer",
            error_message=f"answer for question {index} must not be empty",
            chat_template_error=f"question {index}: model tokenizer must define a chat template",
            context_window_error=(
                f"question {index} and generated answer do not fit within the model context window"
            ),
            json_response=False,
        )
        answers.append(response.root)
    return answers


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


def generate_statements(
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_STATEMENT_PROMPT,
    statement_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str | None = None,
    max_regenerations: int = 0,
) -> list[str]:
    """Generate strict JSON factual statements with a local causal chat model."""
    _validate_max_regenerations(max_regenerations)
    _validate_list_generation_inputs(
        chunk,
        model_name,
        statement_count,
        "statement_count",
        temperature,
        max_new_tokens,
    )
    messages = _build_messages(
        DEFAULT_STATEMENT_SYSTEM_PROMPT,
        prompt,
        {"chunk": chunk, "statement_count": statement_count},
    )
    response = _generate_validated(
        messages,
        _StringListResponse,
        {"expected_count": statement_count},
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device,
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {statement_count} non-empty strings",
        error_message=(
            f"model response must be a JSON array of exactly {statement_count} non-empty strings"
        ),
        chat_template_error="model tokenizer must define a chat template",
        context_window_error=(
            "chunk and generated response do not fit within the model context window"
        ),
        json_response=True,
    )
    return response.root


def generate_information_preservation_statements(
    segment: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str | None = None,
    max_regenerations: int = 0,
) -> tuple[str, list[str]]:
    """Generate one true and three distinct false statements locally."""
    _validate_max_regenerations(max_regenerations)
    _require_non_empty(segment, "segment")
    _require_non_empty(model_name, "model_name")
    _validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=False)
    messages = _build_messages(
        DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
        prompt,
        {"segment": segment},
    )
    response = _generate_validated(
        messages,
        _InformationPreservationResponse,
        None,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device,
        max_regenerations=max_regenerations,
        contract=(
            "a JSON object with one non-empty true_statement and exactly three distinct "
            "non-empty false_statements that differ from true_statement"
        ),
        error_message=(
            "model response must be a JSON object with one non-empty true_statement and exactly "
            "three distinct non-empty false_statements"
        ),
        chat_template_error="model tokenizer must define a chat template",
        context_window_error=(
            "segment and generated response do not fit within the model context window"
        ),
        json_response=True,
    )
    return response.true_statement, response.false_statements


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


def evaluate_information_preservation(
    true_statement: str,
    false_statements: Sequence[str],
    relevant_chunks: Sequence[str],
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    seed: int | None = None,
    device: str | None = None,
    max_regenerations: int = 0,
) -> int:
    """Run one shuffled local HOPE Information Preservation evaluation."""
    _validate_max_regenerations(max_regenerations)
    true_item, false_items, chunk_items = _validate_evaluation_inputs(
        true_statement,
        false_statements,
        relevant_chunks,
        model_name,
        temperature,
        max_new_tokens,
    )
    statements = [(true_item, True), *((statement, False) for statement in false_items)]
    random.Random(seed).shuffle(statements)
    true_index = next(index for index, (_, is_true) in enumerate(statements, 1) if is_true)
    messages = _build_messages(
        DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
        prompt,
        {
            "statements": [
                {"index": index, "statement": statement}
                for index, (statement, _) in enumerate(statements, 1)
            ],
            "relevant_chunks": chunk_items,
        },
    )
    response = _generate_validated(
        messages,
        _InformationPreservationEvaluationResponse,
        None,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device,
        max_regenerations=max_regenerations,
        contract="a JSON object containing only integer selected_index from 1 to 4",
        error_message=(
            "model response must be a JSON object containing only selected_index from 1 to 4"
        ),
        chat_template_error="model tokenizer must define a chat template",
        context_window_error=(
            "statements, relevant chunks, and generated response do not fit within the "
            "model context window"
        ),
        json_response=True,
    )
    return int(response.selected_index == true_index)


def generate_questions(
    chunk: str,
    model_name: str = DEFAULT_STATEMENT_MODEL,
    *,
    prompt: str = DEFAULT_QUESTION_PROMPT,
    question_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    device: str | None = None,
    max_regenerations: int = 0,
) -> list[str]:
    """Generate strict JSON questions with a local causal chat model."""
    _validate_max_regenerations(max_regenerations)
    _validate_list_generation_inputs(
        chunk,
        model_name,
        question_count,
        "question_count",
        temperature,
        max_new_tokens,
    )
    messages = _build_messages(
        DEFAULT_QUESTION_SYSTEM_PROMPT,
        prompt,
        {"chunk": chunk, "question_count": question_count},
    )
    response = _generate_validated(
        messages,
        _StringListResponse,
        {"expected_count": question_count},
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device,
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {question_count} non-empty strings",
        error_message=(
            f"model response must be a JSON array of exactly {question_count} non-empty strings"
        ),
        chat_template_error="model tokenizer must define a chat template",
        context_window_error=(
            "chunk and generated response do not fit within the model context window"
        ),
        json_response=True,
    )
    return response.root
