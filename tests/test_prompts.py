import chunking_metrics
import chunking_metrics.prompts as prompts

PUBLIC_PROMPTS = (
    "DEFAULT_ANSWER_PROMPT",
    "DEFAULT_ANSWER_SYSTEM_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_PROMPT",
    "DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT",
    "DEFAULT_QUESTION_PROMPT",
    "DEFAULT_QUESTION_SYSTEM_PROMPT",
    "DEFAULT_STATEMENT_PROMPT",
    "DEFAULT_STATEMENT_SYSTEM_PROMPT",
)


def test_prompt_constants_are_available_from_package_root() -> None:
    for name in PUBLIC_PROMPTS:
        assert hasattr(prompts, name), f"chunking_metrics.prompts is missing {name}"
        assert getattr(chunking_metrics, name, None) is getattr(prompts, name)
        assert name in chunking_metrics.__all__


def test_information_preservation_prompt_accepts_source_segment() -> None:
    prompt = getattr(prompts, "DEFAULT_INFORMATION_PRESERVATION_PROMPT", None)

    assert prompt is not None
    assert "<source>\nTest segment.\n</source>" in prompt.format(segment="Test segment.")
