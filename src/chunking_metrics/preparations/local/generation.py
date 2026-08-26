import random
from collections.abc import Sequence

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
from . import utils

DEFAULT_STATEMENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
    utils._validate_max_regenerations(max_regenerations)
    question_items, additional_chunk_items = utils._validate_answer_inputs(
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
        response = utils._generate_validated(
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
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_list_generation_inputs(
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
    response = utils._generate_validated(
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
    utils._validate_max_regenerations(max_regenerations)
    utils._require_non_empty(segment, "segment")
    utils._require_non_empty(model_name, "model_name")
    utils._validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=False)
    messages = _build_messages(
        DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
        prompt,
        {"segment": segment},
    )
    response = utils._generate_validated(
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
    utils._validate_max_regenerations(max_regenerations)
    true_item, false_items, chunk_items = utils._validate_evaluation_inputs(
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
    response = utils._generate_validated(
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
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_list_generation_inputs(
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
    response = utils._generate_validated(
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
