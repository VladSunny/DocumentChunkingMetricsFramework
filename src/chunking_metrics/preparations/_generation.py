import json
from string import Formatter

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    ValidationInfo,
    field_validator,
    model_validator,
)


class _NonEmptyTextResponse(RootModel[str]):
    model_config = ConfigDict(strict=True)

    @field_validator("root")
    @classmethod
    def _strip_and_require_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value


class _StringListResponse(RootModel[list[str]]):
    model_config = ConfigDict(strict=True)

    @field_validator("root")
    @classmethod
    def _strip_and_validate_items(
        cls,
        value: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        expected_count = info.context["expected_count"] if info.context else None
        if len(value) != expected_count:
            raise ValueError(f"list must contain exactly {expected_count} items")
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("list items must not be empty")
        return cleaned


class _InformationPreservationResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    true_statement: str
    false_statements: list[str] = Field(min_length=3, max_length=3)

    @field_validator("true_statement")
    @classmethod
    def _strip_true_statement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("true_statement must not be empty")
        return value

    @field_validator("false_statements")
    @classmethod
    def _strip_and_validate_false_statements(cls, value: list[str]) -> list[str]:
        cleaned = [statement.strip() for statement in value]
        if any(not statement for statement in cleaned):
            raise ValueError("false_statements must not contain empty strings")
        if len(set(cleaned)) != 3:
            raise ValueError("false_statements must be distinct")
        return cleaned

    @model_validator(mode="after")
    def _require_false_statements_to_differ_from_true(
        self,
    ) -> "_InformationPreservationResponse":
        if self.true_statement in self.false_statements:
            raise ValueError("false_statements must not contain true_statement")
        return self


class _InformationPreservationEvaluationResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    selected_index: int = Field(ge=1, le=4)


def _build_messages(
    system_prompt: str,
    prompt: str,
    values: dict[str, object],
) -> list[dict[str, str]]:
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

    expected_fields = set(values)
    if prompt_fields != expected_fields:
        missing_fields = expected_fields - prompt_fields
        if missing_fields:
            ordered_placeholders = [f"{{{field}}}" for field in values]
            if len(ordered_placeholders) == 1:
                placeholders = ordered_placeholders[0]
            elif len(ordered_placeholders) == 2:
                placeholders = " and ".join(ordered_placeholders)
            else:
                placeholders = ", ".join(ordered_placeholders[:-1])
                placeholders += f", and {ordered_placeholders[-1]}"
            raise ValueError(f"prompt must contain {placeholders}")
        unsupported_field = sorted(prompt_fields - expected_fields)[0]
        raise ValueError(f"prompt contains an unsupported placeholder: {unsupported_field}")

    try:
        formatted_values = {
            field: value if isinstance(value, int) else json.dumps(value, ensure_ascii=False)
            for field, value in values.items()
        }
        user_prompt = prompt.format(**formatted_values)
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("prompt must be a valid format string") from error
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
