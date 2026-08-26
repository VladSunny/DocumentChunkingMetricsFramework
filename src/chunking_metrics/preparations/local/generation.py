import random
from collections.abc import Sequence
from typing import Any

import torch

from ... import utils
from ...prompts import (
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_STATEMENT_PROMPT,
)
from .calculation import _load_model_and_tokenizer, _resolve_device

DEFAULT_STATEMENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
        max_regenerations: Additional attempts allowed after an invalid model response.

    Returns:
        One stripped, non-empty answer per question, preserving question order.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, a prompt and response budget exceed the context
            window, the tokenizer has no chat template, or an answer is empty.

    Each question is generated independently without truncation. The regeneration limit applies
    separately to each question.
    """
    utils._validate_max_regenerations(max_regenerations)
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
    answers: list[str] = []
    for index, (question, additional_chunks) in enumerate(
        zip(question_items, additional_chunk_items, strict=True)
    ):
        original_messages = utils._answer_messages(question, chunk, additional_chunks, prompt)
        chat_template_error = f"question {index}: model tokenizer must define a chat template"
        context_window_error = (
            f"question {index} and generated answer do not fit within the model context window"
        )

        def generate(
            messages: list[dict[str, str]],
            chat_template_error: str = chat_template_error,
            context_window_error: str = context_window_error,
        ) -> object:
            return _generate_text(
                messages,
                model_name,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                device=device,
                chat_template_error=chat_template_error,
                context_window_error=context_window_error,
            )

        answers.append(
            utils._generate_with_validation(
                original_messages,
                generate,
                lambda response, question_index=index: utils._clean_answer(
                    response, question_index
                ),
                max_regenerations=max_regenerations,
                contract="a non-empty text answer",
                error_message=f"answer for question {index} must not be empty",
            )
        )
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
        max_regenerations: Additional attempts allowed after an invalid model response.

    Returns:
        A list containing exactly ``statement_count`` non-empty statements.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic. The chunk is never truncated.
    """
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_statement_arguments(
        chunk,
        model_name,
        prompt,
        statement_count,
        temperature,
        max_new_tokens,
        device,
    )
    messages = utils._statement_messages(chunk, statement_count, prompt)

    def generate(attempt_messages: list[dict[str, str]]) -> object:
        return _generate_text(
            attempt_messages,
            model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            chat_template_error="model tokenizer must define a chat template",
            context_window_error=(
                "chunk and generated response do not fit within the model context window"
            ),
        )

    error_message = (
        f"model response must be a JSON array of exactly {statement_count} non-empty strings"
    )
    return utils._generate_with_validation(
        messages,
        generate,
        lambda response: utils._parse_statement_response(response, statement_count),
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {statement_count} non-empty strings",
        error_message=error_message,
    )


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
    """Generate one true and three false statements for HOPE Information Preservation.

    Args:
        segment: Source document segment from which the statements are generated.
        model_name: Hugging Face causal chat model identifier or local model path.
        prompt: User-message format string containing the ``{segment}`` placeholder.
        temperature: Positive non-zero sampling temperature.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override.
        max_regenerations: Additional attempts allowed after an invalid model response.

    Returns:
        The stripped true statement and exactly three distinct stripped false statements.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument, context-window budget, chat template, or model response
            violates the required contract.

    Generation does not truncate the segment.
    """
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_local_information_preservation_arguments(
        segment,
        model_name,
        prompt,
        temperature,
        max_new_tokens,
        device,
    )
    messages = utils._information_preservation_messages(segment, prompt)

    def generate(attempt_messages: list[dict[str, str]]) -> object:
        return _generate_text(
            attempt_messages,
            model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            chat_template_error="model tokenizer must define a chat template",
            context_window_error=(
                "segment and generated response do not fit within the model context window"
            ),
        )

    error_message = (
        "model response must be a JSON object with one non-empty true_statement and exactly "
        "three distinct non-empty false_statements"
    )
    return utils._generate_with_validation(
        messages,
        generate,
        utils._parse_information_preservation_response,
        max_regenerations=max_regenerations,
        contract=(
            "a JSON object with one non-empty true_statement and exactly three distinct "
            "non-empty false_statements that differ from true_statement"
        ),
        error_message=error_message,
    )


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
    """Run one local HOPE Information Preservation multiple-choice evaluation.

    Args:
        true_statement: One non-empty statement known to be true.
        false_statements: Exactly three distinct non-empty false statements.
        relevant_chunks: Non-empty chunks retrieved as evidence for the true statement.
        model_name: Hugging Face causal chat model identifier or local model path.
        prompt: User-message format string containing ``{statements}`` and
            ``{relevant_chunks}``. Both values are inserted as JSON.
        temperature: Non-negative generation temperature. Zero selects greedy decoding.
        max_new_tokens: Maximum number of tokens available for the JSON response.
        seed: Optional seed used only to shuffle the four statements reproducibly.
        device: Optional ``cpu``, ``cuda[:index]``, or ``mps`` override.
        max_regenerations: Additional attempts allowed after an invalid model response.

    Returns:
        ``1`` when the model selects the true statement, otherwise ``0``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument, context-window budget, chat template, or model response
            violates the required contract.

    Evaluation shuffles statements once, does not truncate inputs, and does not mutate them.
    """
    utils._validate_max_regenerations(max_regenerations)
    true_statement_item, false_statement_items, relevant_chunk_items = (
        utils._validate_local_information_preservation_evaluation_arguments(
            true_statement,
            false_statements,
            relevant_chunks,
            model_name,
            prompt,
            temperature,
            max_new_tokens,
            seed,
            device,
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
    messages = utils._information_preservation_evaluation_messages(
        [statement for statement, _ in statements],
        relevant_chunk_items,
        prompt,
    )

    def generate(attempt_messages: list[dict[str, str]]) -> object:
        return _generate_text(
            attempt_messages,
            model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            chat_template_error="model tokenizer must define a chat template",
            context_window_error=(
                "statements, relevant chunks, and generated response do not fit within the "
                "model context window"
            ),
        )

    selected_index = utils._generate_with_validation(
        messages,
        generate,
        utils._parse_information_preservation_evaluation_response,
        max_regenerations=max_regenerations,
        contract="a JSON object containing only integer selected_index from 1 to 4",
        error_message=(
            "model response must be a JSON object containing only selected_index from 1 to 4"
        ),
    )
    return int(selected_index == true_statement_index)


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
        max_regenerations: Additional attempts allowed after an invalid model response.

    Returns:
        A list containing exactly ``question_count`` non-empty questions.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument is invalid, the prompt and response budget do not fit the
            context window, the tokenizer has no chat template, or the model response is not a
            JSON array containing exactly the requested number of non-empty strings.

    Generation is stochastic. The chunk is never truncated.
    """
    utils._validate_max_regenerations(max_regenerations)
    utils._validate_question_arguments(
        chunk,
        model_name,
        prompt,
        question_count,
        temperature,
        max_new_tokens,
        device,
    )
    messages = utils._question_messages(chunk, question_count, prompt)

    def generate(attempt_messages: list[dict[str, str]]) -> object:
        return _generate_text(
            attempt_messages,
            model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            device=device,
            chat_template_error="model tokenizer must define a chat template",
            context_window_error=(
                "chunk and generated response do not fit within the model context window"
            ),
        )

    error_message = (
        f"model response must be a JSON array of exactly {question_count} non-empty strings"
    )
    return utils._generate_with_validation(
        messages,
        generate,
        lambda response: utils._parse_question_response(response, question_count),
        max_regenerations=max_regenerations,
        contract=f"a JSON array of exactly {question_count} non-empty strings",
        error_message=error_message,
    )
