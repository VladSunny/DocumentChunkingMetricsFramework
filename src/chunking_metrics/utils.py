import math
from collections.abc import Sequence
from string import Formatter

import numpy as np


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


def _validate_information_preservation_arguments(
    segment: str,
    model_name: str,
    api_key: str,
    base_url: str | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> None:
    if not isinstance(segment, str):
        raise TypeError("segment must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if not segment.strip():
        raise ValueError("segment must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if base_url is not None and not base_url.strip():
        raise ValueError("base_url must not be empty")
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
    if "segment" not in prompt_fields:
        raise ValueError("prompt must contain {segment}")
    unsupported_fields = prompt_fields - {"segment"}
    if unsupported_fields:
        unsupported_field = sorted(unsupported_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")


def _validate_information_preservation_evaluation_arguments(
    true_statement: str,
    false_statements: Sequence[str],
    relevant_chunks: Sequence[str],
    model_name: str,
    api_key: str,
    base_url: str | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    seed: int | None,
) -> tuple[str, list[str], list[str]]:
    if not isinstance(true_statement, str):
        raise TypeError("true_statement must be a string")
    if isinstance(false_statements, (str, bytes)) or not isinstance(false_statements, Sequence):
        raise TypeError("false_statements must be a sequence of strings")
    false_statement_items = list(false_statements)
    for index, statement in enumerate(false_statement_items):
        if not isinstance(statement, str):
            raise TypeError(f"false_statements[{index}] must be a string")
    if isinstance(relevant_chunks, (str, bytes)) or not isinstance(relevant_chunks, Sequence):
        raise TypeError("relevant_chunks must be a sequence of strings")
    relevant_chunk_items = list(relevant_chunks)
    for index, chunk in enumerate(relevant_chunk_items):
        if not isinstance(chunk, str):
            raise TypeError(f"relevant_chunks[{index}] must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(api_key, str):
        raise TypeError("api_key must be a string")
    if base_url is not None and not isinstance(base_url, str):
        raise TypeError("base_url must be a string or None")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")

    true_statement_item = true_statement.strip()
    if not true_statement_item:
        raise ValueError("true_statement must not be empty")
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
    if not relevant_chunk_items:
        raise ValueError("relevant_chunks must not be empty")
    for index, chunk in enumerate(relevant_chunk_items):
        if not chunk.strip():
            raise ValueError(f"relevant_chunks[{index}] must not be empty")
    relevant_chunk_items = [chunk.strip() for chunk in relevant_chunk_items]
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not api_key.strip():
        raise ValueError("api_key must not be empty")
    if base_url is not None and not base_url.strip():
        raise ValueError("base_url must not be empty")
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
    required_fields = {"statements", "relevant_chunks"}
    if not required_fields.issubset(prompt_fields):
        raise ValueError("prompt must contain {statements} and {relevant_chunks}")
    unsupported_fields = prompt_fields - required_fields
    if unsupported_fields:
        unsupported_field = sorted(unsupported_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature < 0:
        raise ValueError("temperature must be greater than or equal to zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    return true_statement_item, false_statement_items, relevant_chunk_items


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


def _validate_answer_arguments(
    questions: Sequence[str],
    chunk: str,
    model_name: str,
    additional_chunks_by_question: Sequence[Sequence[str]] | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    device: str | None,
) -> tuple[list[str], list[list[str]]]:
    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be a sequence of strings")
    question_items = list(questions)
    for index, question in enumerate(question_items):
        if not isinstance(question, str):
            raise TypeError(f"questions[{index}] must be a string")
    if not isinstance(chunk, str):
        raise TypeError("chunk must be a string")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a number")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an integer")
    if device is not None and not isinstance(device, str):
        raise TypeError("device must be a string or None")
    if additional_chunks_by_question is not None and (
        isinstance(additional_chunks_by_question, (str, bytes))
        or not isinstance(additional_chunks_by_question, Sequence)
    ):
        raise TypeError(
            "additional_chunks_by_question must be a sequence of string sequences or None"
        )

    if not question_items:
        raise ValueError("questions must not be empty")
    for index, question in enumerate(question_items):
        if not question.strip():
            raise ValueError(f"questions[{index}] must not be empty")
    if not chunk.strip():
        raise ValueError("chunk must not be empty")
    if not model_name.strip():
        raise ValueError("model_name must not be empty")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    if additional_chunks_by_question is None:
        additional_chunk_items = [[] for _ in question_items]
    else:
        if len(additional_chunks_by_question) != len(question_items):
            raise ValueError("additional_chunks_by_question must contain one item per question")
        additional_chunk_items = []
        for question_index, additional_chunks in enumerate(additional_chunks_by_question):
            if isinstance(additional_chunks, (str, bytes)) or not isinstance(
                additional_chunks, Sequence
            ):
                raise TypeError(
                    f"additional_chunks_by_question[{question_index}] must be a sequence of strings"
                )
            chunk_items = list(additional_chunks)
            for chunk_index, additional_chunk in enumerate(chunk_items):
                if not isinstance(additional_chunk, str):
                    raise TypeError(
                        f"additional_chunks_by_question[{question_index}]"
                        f"[{chunk_index}] must be a string"
                    )
                if not additional_chunk.strip():
                    raise ValueError(
                        f"additional_chunks_by_question[{question_index}]"
                        f"[{chunk_index}] must not be empty"
                    )
            additional_chunk_items.append(chunk_items)

    try:
        prompt_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(prompt)
            if field_name is not None
        }
    except ValueError as error:
        raise ValueError("prompt must be a valid format string") from error
    required_fields = {"question", "chunk", "additional_chunks"}
    if not required_fields.issubset(prompt_fields):
        raise ValueError("prompt must contain {question}, {chunk}, and {additional_chunks}")
    unsupported_fields = prompt_fields - required_fields
    if unsupported_fields:
        unsupported_field = sorted(unsupported_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature < 0:
        raise ValueError("temperature must be greater than or equal to zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    return question_items, additional_chunk_items


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


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray | np.float64:
    """Returns cosine similarity along the last axis."""
    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator
