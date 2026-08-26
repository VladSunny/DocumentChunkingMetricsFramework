import json
import math
from collections.abc import Sequence
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

_Response = TypeVar("_Response", bound=BaseModel)


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


def _validate_api_credentials(
    model_name: str,
    api_key: str,
    base_url: str | None,
    *,
    allow_empty_base_url: bool,
) -> None:
    _require_non_empty(model_name, "model_name")
    _require_non_empty(api_key, "api_key")
    if not allow_empty_base_url and base_url is not None:
        _require_non_empty(base_url, "base_url")


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
    client: OpenAI,
    original_messages: list[dict[str, str]],
    response_schema: type[_Response],
    validation_context: dict[str, object] | None,
    *,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    max_regenerations: int,
    contract: str,
    error_message: str,
    json_response: bool,
) -> _Response:
    messages = original_messages
    for attempt in range(max_regenerations + 1):
        request: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if json_response:
            request.update(
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
            )
        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content
        try:
            if json_response:
                return response_schema.model_validate_json(content, context=validation_context)
            return response_schema.model_validate(content, context=validation_context)
        except ValidationError as error:
            if attempt == max_regenerations:
                raise ValueError(error_message) from error
            messages = _regeneration_messages(original_messages, content, contract)
    raise RuntimeError("unreachable")


def _validate_answer_inputs(
    questions: Sequence[str],
    chunk: str,
    model_name: str,
    api_key: str,
    base_url: str | None,
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
    _validate_api_credentials(model_name, api_key, base_url, allow_empty_base_url=False)
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
    api_key: str,
    base_url: str,
    count: int,
    count_name: str,
    temperature: float,
    max_new_tokens: int,
) -> None:
    _require_non_empty(chunk, "chunk")
    _validate_api_credentials(model_name, api_key, base_url, allow_empty_base_url=True)
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
    api_key: str,
    base_url: str | None,
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
    _validate_api_credentials(model_name, api_key, base_url, allow_empty_base_url=False)
    _validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=True)
    return true_statement_item, false_statement_items, relevant_chunk_items
