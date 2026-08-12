import chunking_metrics.prompts as prompts

PUBLIC_PROMPTS = (
    "DEFAULT_ANSWER_PROMPT",
    "DEFAULT_ANSWER_SYSTEM_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT",
    "DEFAULT_QUESTION_PROMPT",
    "DEFAULT_QUESTION_SYSTEM_PROMPT",
    "DEFAULT_STATEMENT_PROMPT",
    "DEFAULT_STATEMENT_SYSTEM_PROMPT",
)


def test_prompt_constants_are_available_from_prompts_module() -> None:
    for name in PUBLIC_PROMPTS:
        assert hasattr(prompts, name), f"chunking_metrics.prompts is missing {name}"


def test_information_preservation_prompt_accepts_source_segment() -> None:
    prompt = getattr(prompts, "DEFAULT_INFORMATION_PRESERVATION_PROMPT", None)

    assert prompt is not None
    assert "<source>\nTest segment.\n</source>" in prompt.format(segment="Test segment.")


def test_information_preservation_evaluation_prompt_accepts_json_inputs() -> None:
    prompt = getattr(prompts, "DEFAULT_INFORMATION_PRESERVATION_EVALUATION_PROMPT", None)

    assert prompt is not None
    formatted = prompt.format(
        statements='[{"index":1,"statement":"True."}]',
        relevant_chunks='["Relevant chunk."]',
    )
    assert '<statements>\n[{"index":1,"statement":"True."}]\n</statements>' in formatted
    assert '<relevant_chunks>\n["Relevant chunk."]\n</relevant_chunks>' in formatted
