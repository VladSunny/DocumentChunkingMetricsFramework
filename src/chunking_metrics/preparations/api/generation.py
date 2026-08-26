import random
from collections.abc import Sequence

from openai import OpenAI

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


def generate_answers(
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
    max_regenerations: int = 0,
) -> list[str]:
    """Answer questions independently through OpenAI-compatible Chat Completions."""
    utils._validate_max_regenerations(max_regenerations)
    question_items, additional_chunk_items = utils._validate_answer_inputs(
        questions,
        chunk,
        model_name,
        api_key,
        base_url,
        additional_chunks_by_question,
        temperature,
        max_new_tokens,
    )
    message_items = [
        _build_messages(
            DEFAULT_ANSWER_SYSTEM_PROMPT,
            prompt,
            {
                "question": question,
                "chunk": chunk,
                "additional_chunks": additional_chunks,
            },
        )
        for question, additional_chunks in zip(question_items, additional_chunk_items, strict=True)
    ]
    client = OpenAI(api_key=api_key, base_url=base_url)
    answers: list[str] = []
    for index, messages in enumerate(message_items):
        answer = utils._generate_validated(
            client,
            messages,
            _NonEmptyTextResponse,
            None,
            model_name=model_name.strip(),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_regenerations=max_regenerations,
            contract="a non-empty text answer",
            error_message=f"answer for question {index} must not be empty",
            json_response=False,
        )
        answers.append(answer.root)
    return answers


def generate_statements(
    chunk: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    *,
    prompt: str = DEFAULT_STATEMENT_PROMPT,
    statement_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_regenerations: int = 0,
) -> list[str]:
    """Generate a strict JSON list of factual statements through an API."""
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_list_generation_inputs(
        chunk,
        model_name,
        api_key,
        base_url,
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
        OpenAI(api_key=api_key, base_url=base_url),
        messages,
        _StringListResponse,
        {"expected_count": statement_count},
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {statement_count} non-empty strings",
        error_message=(
            f"model response must be a JSON array of exactly {statement_count} non-empty strings"
        ),
        json_response=True,
    )
    return response.root


def generate_information_preservation_statements(
    segment: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str | None = None,
    *,
    prompt: str = DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_regenerations: int = 0,
) -> tuple[str, list[str]]:
    """Generate one true and three distinct false statements through an API."""
    utils._validate_max_regenerations(max_regenerations)
    utils._require_non_empty(segment, "segment")
    utils._validate_api_credentials(model_name, api_key, base_url, allow_empty_base_url=False)
    utils._validate_generation_limits(temperature, max_new_tokens, allow_zero_temperature=False)
    messages = _build_messages(
        DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
        prompt,
        {"segment": segment},
    )
    response = utils._generate_validated(
        OpenAI(api_key=api_key, base_url=base_url),
        messages,
        _InformationPreservationResponse,
        None,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_regenerations=max_regenerations,
        contract=(
            "a JSON object with one non-empty true_statement and exactly three distinct "
            "non-empty false_statements that differ from true_statement"
        ),
        error_message=(
            "model response must be a JSON object with one non-empty true_statement and exactly "
            "three distinct non-empty false_statements"
        ),
        json_response=True,
    )
    return response.true_statement, response.false_statements


def evaluate_information_preservation(
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
    max_regenerations: int = 0,
) -> int:
    """Run one shuffled HOPE Information Preservation evaluation through an API."""
    utils._validate_max_regenerations(max_regenerations)
    true_item, false_items, chunk_items = utils._validate_evaluation_inputs(
        true_statement,
        false_statements,
        relevant_chunks,
        model_name,
        api_key,
        base_url,
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
        OpenAI(api_key=api_key, base_url=base_url),
        messages,
        _InformationPreservationEvaluationResponse,
        None,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_regenerations=max_regenerations,
        contract="a JSON object containing only integer selected_index from 1 to 4",
        error_message=(
            "model response must be a JSON object containing only selected_index from 1 to 4"
        ),
        json_response=True,
    )
    return int(response.selected_index == true_index)


def generate_questions(
    chunk: str,
    model_name: str = "",
    api_key: str = "",
    base_url: str = "",
    *,
    prompt: str = DEFAULT_QUESTION_PROMPT,
    question_count: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_regenerations: int = 0,
) -> list[str]:
    """Generate a strict JSON list of answerable questions through an API."""
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_list_generation_inputs(
        chunk,
        model_name,
        api_key,
        base_url,
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
        OpenAI(api_key=api_key, base_url=base_url),
        messages,
        _StringListResponse,
        {"expected_count": question_count},
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {question_count} non-empty strings",
        error_message=(
            f"model response must be a JSON array of exactly {question_count} non-empty strings"
        ),
        json_response=True,
    )
    return response.root
