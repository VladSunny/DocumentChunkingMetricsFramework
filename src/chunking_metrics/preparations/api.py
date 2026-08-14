import random
from collections.abc import Sequence

from openai import OpenAI

from .. import utils
from ..prompts import (
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT,
    DEFAULT_INFORMATION_PRESERVATION_PROMPT,
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_STATEMENT_PROMPT,
)


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
            messages=utils._answer_messages(question, chunk, additional_chunks, prompt),
            stream=False,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        answers.append(utils._clean_answer(response.choices[0].message.content, index))
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
    messages = utils._statement_messages(chunk, statement_count, prompt)

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

    return utils._parse_statement_response(content, statement_count)


def generate_information_preservation_statements(
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
    messages = utils._information_preservation_messages(segment, prompt)
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
    return utils._parse_information_preservation_response(content)


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
    messages = utils._information_preservation_evaluation_messages(
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
    selected_index = utils._parse_information_preservation_evaluation_response(
        response.choices[0].message.content
    )
    return int(selected_index == true_statement_index)


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

    messages = utils._question_messages(chunk, question_count, prompt)

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

    return utils._parse_question_response(content, question_count)
