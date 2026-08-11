import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import chunking_metrics
from chunking_metrics import calculate_perplexity, preparation


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 1

    def __init__(self, model_max_length: int = 32) -> None:
        self.model_max_length = model_max_length

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class FakeModel:
    def __init__(self, max_position_embeddings: int = 32, loss: float = math.log(4.0)) -> None:
        self.config = SimpleNamespace(
            bos_token_id=0,
            eos_token_id=1,
            max_position_embeddings=max_position_embeddings,
        )
        self.loss = loss
        self.input_ids: list[int] | None = None
        self.labels: list[int] | None = None
        self.loaded_device: str | None = None
        self.is_eval = False

    def to(self, device: str) -> "FakeModel":
        self.loaded_device = device
        return self

    def eval(self) -> "FakeModel":
        self.is_eval = True
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.input_ids = kwargs["input_ids"].tolist()[0]
        self.labels = kwargs["labels"].tolist()[0]
        return SimpleNamespace(loss=torch.tensor(self.loss))


class FakeEmbeddingTokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is True
        assert padding is False
        assert truncation is False
        return {"input_ids": [[0, *range(len(text)), 1] for text in texts]}


class FakeEmbeddingModel:
    def __init__(
        self,
        max_seq_length: int = 32,
        embedding_dtype: np.dtype[Any] | None = None,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.embedding_dtype = embedding_dtype or np.dtype(np.float32)
        self.tokenizer = FakeEmbeddingTokenizer()
        self.is_eval = False

    def eval(self) -> "FakeEmbeddingModel":
        self.is_eval = True
        return self

    def encode(self, texts: str | list[str], **kwargs: Any) -> np.ndarray:
        assert kwargs == {
            "batch_size": 4,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if isinstance(texts, str):
            return np.array([0.6, 0.8], dtype=self.embedding_dtype)
        assert isinstance(texts, list)
        return np.array([[0.6, 0.8] for _ in texts], dtype=self.embedding_dtype)


class FakeStatementTokenizer:
    eos_token_id = 2
    pad_token_id = 3

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None
        self.decoded_ids: list[int] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> dict[str, torch.Tensor]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        assert return_dict is True
        self.messages = messages
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids = token_ids.tolist()
        return self.response


class FakeStatementModel:
    def __init__(self) -> None:
        self.generation_arguments: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_arguments = kwargs
        return torch.tensor([[10, 11, 20, 21]], dtype=torch.long)


def install_fake_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_length: int = 32,
) -> FakeModel:
    tokenizer = FakeTokenizer(model_max_length=max_length)
    model = FakeModel(max_position_embeddings=max_length)
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, max_length),
    )
    return model


def test_generate_statements_returns_exact_cleaned_json_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('[" Первое утверждение. ", "Второе утверждение."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = preparation.generate_statements(
        "Текст чанка.",
        model_name="model",
        statement_count=2,
        temperature=0.5,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первое утверждение.", "Второе утверждение."]
    assert tokenizer.decoded_ids == [20, 21]
    assert tokenizer.messages is not None
    assert "Текст чанка." in tokenizer.messages[-1]["content"]
    assert "2" in tokenizer.messages[-1]["content"]
    assert model.generation_arguments is not None
    assert model.generation_arguments["do_sample"] is True
    assert model.generation_arguments["temperature"] == 0.5
    assert model.generation_arguments["max_new_tokens"] == 8


def test_generate_statements_formats_custom_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = preparation.generate_statements(
        "Текст чанка.",
        model_name="model",
        prompt="Create {statement_count} claims from {chunk}.",
        statement_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первое.", "Второе."]
    assert tokenizer.messages is not None
    assert tokenizer.messages[-1] == {
        "role": "user",
        "content": 'Create 2 claims from "Текст чанка.".',
    }


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("chunk", 123, "chunk must be a string"),
        ("model_name", 123, "model_name must be a string"),
        ("prompt", 123, "prompt must be a string"),
        ("statement_count", 1.5, "statement_count must be an integer"),
        ("statement_count", True, "statement_count must be an integer"),
        ("temperature", "hot", "temperature must be a number"),
        ("temperature", True, "temperature must be a number"),
        ("max_new_tokens", 1.5, "max_new_tokens must be an integer"),
        ("max_new_tokens", True, "max_new_tokens must be an integer"),
        ("device", 123, "device must be a string or None"),
    ],
)
def test_generate_statements_rejects_invalid_argument_types_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    arguments: dict[str, object] = {
        "chunk": "Текст чанка.",
        "model_name": "model",
        "statement_count": 2,
        "temperature": 0.5,
        "max_new_tokens": 8,
        "device": "cpu",
        argument: value,
    }

    with pytest.raises(TypeError, match=message):
        preparation.generate_statements(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"chunk": "  "}, "chunk must not be empty"),
        ({"model_name": "  "}, "model_name must not be empty"),
        ({"prompt": "  "}, "prompt must not be empty"),
        (
            {"prompt": "Generate {statement_count} statements."},
            r"prompt must contain \{chunk\} and \{statement_count\}",
        ),
        (
            {"prompt": "Generate statements from {chunk}."},
            r"prompt must contain \{chunk\} and \{statement_count\}",
        ),
        (
            {"prompt": "Generate {statement_count} statements from {chunk}: {unknown}"},
            "prompt contains an unsupported placeholder: unknown",
        ),
        ({"statement_count": 0}, "statement_count must be greater than zero"),
        ({"statement_count": -1}, "statement_count must be greater than zero"),
        ({"temperature": 0}, "temperature must be greater than zero"),
        ({"temperature": -0.1}, "temperature must be greater than zero"),
        ({"temperature": float("nan")}, "temperature must be finite"),
        ({"temperature": float("inf")}, "temperature must be finite"),
        ({"max_new_tokens": 0}, "max_new_tokens must be greater than zero"),
        ({"max_new_tokens": -1}, "max_new_tokens must be greater than zero"),
    ],
)
def test_generate_statements_rejects_invalid_argument_values_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    call_arguments: dict[str, object] = {
        "chunk": "Текст чанка.",
        "model_name": "model",
        "statement_count": 2,
        "temperature": 0.5,
        "max_new_tokens": 8,
        "device": "cpu",
        **arguments,
    }

    with pytest.raises(ValueError, match=message):
        preparation.generate_statements(**call_arguments)  # type: ignore[arg-type]


def test_generate_statements_rejects_chunk_that_cannot_fit_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 9),
    )

    with pytest.raises(ValueError, match="do not fit within the model context window"):
        preparation.generate_statements(
            "Текст чанка.",
            model_name="model",
            statement_count=2,
            max_new_tokens=8,
            device="cpu",
        )

    assert model.generation_arguments is None


def test_generate_statements_requires_tokenizer_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()

    def reject_chat_template(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Cannot use chat template functions")

    tokenizer.apply_chat_template = reject_chat_template  # type: ignore[method-assign]
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match="tokenizer must define a chat template"):
        preparation.generate_statements("Текст чанка.", model_name="model", device="cpu")


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"statement": "Первое."}',
        '["Только одно."]',
        '["Первое.", 2]',
        '["Первое.", "  "]',
    ],
)
def test_generate_statements_rejects_response_outside_strict_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    tokenizer = FakeStatementTokenizer(response)
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(
        ValueError,
        match="model response must be a JSON array of exactly 2 non-empty strings",
    ):
        preparation.generate_statements(
            "Текст чанка.",
            model_name="model",
            statement_count=2,
            max_new_tokens=8,
            device="cpu",
        )


def test_generate_statements_is_exported_from_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = chunking_metrics.generate_statements(
        "Текст чанка.",
        model_name="model",
        statement_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первое.", "Второе."]
    assert chunking_metrics.DEFAULT_STATEMENT_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_generate_questions_returns_exact_cleaned_json_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('[" Первый вопрос? ", "Второй вопрос?"]')
    model = FakeStatementModel()

    def load_model(model_name: str, device: str) -> tuple[Any, Any, int]:
        assert model_name == preparation.DEFAULT_STATEMENT_MODEL
        assert device == "cpu"
        return tokenizer, model, 32

    monkeypatch.setattr(preparation, "_load_model_and_tokenizer", load_model)

    result = preparation.generate_questions(
        "Текст чанка.",
        question_count=2,
        temperature=0.5,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первый вопрос?", "Второй вопрос?"]
    assert tokenizer.decoded_ids == [20, 21]
    assert tokenizer.messages is not None
    assert "Текст чанка." in tokenizer.messages[-1]["content"]
    assert "2" in tokenizer.messages[-1]["content"]
    assert model.generation_arguments is not None
    assert model.generation_arguments["do_sample"] is True
    assert model.generation_arguments["temperature"] == 0.5
    assert model.generation_arguments["max_new_tokens"] == 8


def test_generate_questions_formats_custom_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = preparation.generate_questions(
        "Текст чанка.",
        model_name="model",
        prompt="Create {question_count} questions from {chunk}.",
        question_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первый?", "Второй?"]
    assert tokenizer.messages is not None
    assert tokenizer.messages[-1] == {
        "role": "user",
        "content": 'Create 2 questions from "Текст чанка.".',
    }


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("chunk", 123, "chunk must be a string"),
        ("model_name", 123, "model_name must be a string"),
        ("prompt", 123, "prompt must be a string"),
        ("question_count", 1.5, "question_count must be an integer"),
        ("question_count", True, "question_count must be an integer"),
        ("temperature", "hot", "temperature must be a number"),
        ("temperature", True, "temperature must be a number"),
        ("max_new_tokens", 1.5, "max_new_tokens must be an integer"),
        ("max_new_tokens", True, "max_new_tokens must be an integer"),
        ("device", 123, "device must be a string or None"),
    ],
)
def test_generate_questions_rejects_invalid_argument_types_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    arguments: dict[str, object] = {
        "chunk": "Текст чанка.",
        "model_name": "model",
        "question_count": 2,
        "temperature": 0.5,
        "max_new_tokens": 8,
        "device": "cpu",
        argument: value,
    }

    with pytest.raises(TypeError, match=message):
        preparation.generate_questions(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"chunk": "  "}, "chunk must not be empty"),
        ({"model_name": "  "}, "model_name must not be empty"),
        ({"prompt": "  "}, "prompt must not be empty"),
        (
            {"prompt": "Generate {question_count} questions."},
            r"prompt must contain \{chunk\} and \{question_count\}",
        ),
        (
            {"prompt": "Generate questions from {chunk}."},
            r"prompt must contain \{chunk\} and \{question_count\}",
        ),
        (
            {"prompt": "Generate {question_count} questions from {chunk}: {unknown}"},
            "prompt contains an unsupported placeholder: unknown",
        ),
        ({"question_count": 0}, "question_count must be greater than zero"),
        ({"question_count": -1}, "question_count must be greater than zero"),
        ({"temperature": 0}, "temperature must be greater than zero"),
        ({"temperature": -0.1}, "temperature must be greater than zero"),
        ({"temperature": float("nan")}, "temperature must be finite"),
        ({"temperature": float("inf")}, "temperature must be finite"),
        ({"max_new_tokens": 0}, "max_new_tokens must be greater than zero"),
        ({"max_new_tokens": -1}, "max_new_tokens must be greater than zero"),
    ],
)
def test_generate_questions_rejects_invalid_argument_values_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    call_arguments: dict[str, object] = {
        "chunk": "Текст чанка.",
        "model_name": "model",
        "question_count": 2,
        "temperature": 0.5,
        "max_new_tokens": 8,
        "device": "cpu",
        **arguments,
    }

    with pytest.raises(ValueError, match=message):
        preparation.generate_questions(**call_arguments)  # type: ignore[arg-type]


def test_generate_questions_rejects_chunk_that_cannot_fit_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 9),
    )

    with pytest.raises(ValueError, match="do not fit within the model context window"):
        preparation.generate_questions(
            "Текст чанка.",
            model_name="model",
            question_count=2,
            max_new_tokens=8,
            device="cpu",
        )

    assert model.generation_arguments is None


def test_generate_questions_requires_tokenizer_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()

    def reject_chat_template(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Cannot use chat template functions")

    tokenizer.apply_chat_template = reject_chat_template  # type: ignore[method-assign]
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match="tokenizer must define a chat template"):
        preparation.generate_questions("Текст чанка.", model_name="model", device="cpu")


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"question": "Первый?"}',
        '["Только один?"]',
        '["Первый?", 2]',
        '["Первый?", "  "]',
    ],
)
def test_generate_questions_rejects_response_outside_strict_json_contract(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    tokenizer = FakeStatementTokenizer(response)
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(
        ValueError,
        match="model response must be a JSON array of exactly 2 non-empty strings",
    ):
        preparation.generate_questions(
            "Текст чанка.",
            model_name="model",
            question_count=2,
            max_new_tokens=8,
            device="cpu",
        )


def test_generate_questions_is_exported_from_package(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        preparation,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = chunking_metrics.generate_questions(
        "Текст чанка.",
        model_name="model",
        question_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первый?", "Второй?"]


def test_calculate_embeddings_returns_vector_for_single_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: model,
        raising=False,
    )

    result = preparation.calculate_embeddings(
        "text",
        model_name="model",
        device="cpu",
        batch_size=4,
    )

    np.testing.assert_array_equal(result, np.array([0.6, 0.8], dtype=np.float32))


def test_calculate_embeddings_returns_float32_matrix_for_text_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel(embedding_dtype=np.dtype(np.float64))
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: model,
        raising=False,
    )

    result = preparation.calculate_embeddings(
        ("first", "second"),
        model_name="model",
        device="cpu",
        batch_size=4,
    )

    assert result.shape == (2, 2)
    assert result.dtype == np.float32


def test_embedding_model_loader_caches_model_and_enables_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel()
    loads: list[tuple[str, str]] = []

    def load_model(model_name: str, *, device: str) -> FakeEmbeddingModel:
        loads.append((model_name, device))
        return model

    monkeypatch.setattr(preparation, "SentenceTransformer", load_model, raising=False)
    preparation._load_embedding_model.cache_clear()

    first = preparation._load_embedding_model("model", "cpu")
    second = preparation._load_embedding_model("model", "cpu")

    assert first is second
    assert loads == [("model", "cpu")]
    assert model.is_eval
    preparation._load_embedding_model.cache_clear()


def test_calculate_embeddings_rejects_empty_text_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )

    with pytest.raises(ValueError, match="texts must not be empty"):
        preparation.calculate_embeddings([], device="cpu")


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("texts", 123, "texts must be a string or a sequence of strings"),
        ("texts", ["text", 123], r"texts\[1\] must be a string"),
        ("model_name", 123, "model_name must be a string"),
        ("device", 123, "device must be a string or None"),
        ("batch_size", 1.5, "batch_size must be an integer"),
        ("batch_size", True, "batch_size must be an integer"),
    ],
)
def test_calculate_embeddings_rejects_invalid_argument_types(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    arguments: dict[str, object] = {
        "texts": "text",
        "model_name": "model",
        "device": "cpu",
        "batch_size": 4,
        argument: value,
    }

    with pytest.raises(TypeError, match=message):
        preparation.calculate_embeddings(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"texts": "  "}, r"texts\[0\] must not be empty"),
        ({"texts": ["text", "\t"]}, r"texts\[1\] must not be empty"),
        ({"texts": "text", "model_name": "  "}, "model_name must not be empty"),
        ({"texts": "text", "batch_size": 0}, "batch_size must be greater than zero"),
        ({"texts": "text", "batch_size": -1}, "batch_size must be greater than zero"),
    ],
)
def test_calculate_embeddings_rejects_invalid_argument_values(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )

    with pytest.raises(ValueError, match=message):
        preparation.calculate_embeddings(**arguments, device="cpu")  # type: ignore[arg-type]


def test_calculate_embeddings_warns_when_texts_are_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel(max_seq_length=5)
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: model,
    )

    with pytest.warns(
        UserWarning,
        match="2 texts exceed the model limit of 5 tokens and will be truncated",
    ):
        result = preparation.calculate_embeddings(
            ["abcd", "x", "abcdef"],
            model_name="model",
            device="cpu",
            batch_size=4,
        )

    assert result.shape == (3, 2)


def test_calculate_embeddings_is_exported_from_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(
        preparation,
        "_load_embedding_model",
        lambda model_name, device: model,
    )

    result = chunking_metrics.calculate_embeddings(
        "text",
        model_name="model",
        device="cpu",
        batch_size=4,
    )

    assert result.shape == (2,)


def test_calculate_perplexity_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        calculate_perplexity("")


def test_calculate_perplexity_scores_only_normalized_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = install_fake_components(monkeypatch)

    result = calculate_perplexity("  text", context="abc ", device="cpu")

    target_ids = [ord(character) for character in " text"]
    assert result == pytest.approx(4.0)
    assert model.input_ids == [0, ord("a"), ord("b"), ord("c"), *target_ids]
    assert model.labels == [-100, -100, -100, -100, *target_ids]


def test_calculate_perplexity_warns_when_context_is_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = install_fake_components(monkeypatch, max_length=6)

    with pytest.warns(UserWarning, match="discarded 3 context tokens"):
        calculate_perplexity("x", context="abcdef", device="cpu")

    target_ids = [ord(" "), ord("x")]
    assert model.input_ids == [0, ord("d"), ord("e"), ord("f"), *target_ids]
    assert model.labels == [-100, -100, -100, -100, *target_ids]


def test_calculate_perplexity_rejects_target_larger_than_model_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_components(monkeypatch, max_length=4)

    with pytest.raises(ValueError, match="target requires 5 tokens"):
        calculate_perplexity("abc", device="cpu")


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("text", 123, "text must be a string"),
        ("model_name", 123, "model_name must be a string"),
        ("context", 123, "context must be a string or None"),
        ("device", 123, "device must be a string or None"),
    ],
)
def test_calculate_perplexity_rejects_invalid_argument_types(
    argument: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {"text": "text", argument: value}

    with pytest.raises(TypeError, match=message):
        calculate_perplexity(**arguments)  # type: ignore[arg-type]


def test_calculate_perplexity_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name must not be empty"):
        calculate_perplexity("text", model_name="  ")


def test_resolve_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert preparation._resolve_device(None) == "cuda"


def test_resolve_device_uses_mps_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert preparation._resolve_device(None) == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert preparation._resolve_device(None) == "cpu"


def test_model_loader_caches_last_model(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer(model_max_length=64)
    model = FakeModel(max_position_embeddings=32)
    tokenizer_loads = 0
    model_loads = 0

    def load_tokenizer(model_name: str) -> FakeTokenizer:
        nonlocal tokenizer_loads
        tokenizer_loads += 1
        return tokenizer

    def load_model(model_name: str) -> FakeModel:
        nonlocal model_loads
        model_loads += 1
        return model

    monkeypatch.setattr(preparation.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(preparation.AutoModelForCausalLM, "from_pretrained", load_model)
    preparation._load_model_and_tokenizer.cache_clear()

    first = preparation._load_model_and_tokenizer("model", "cpu")
    second = preparation._load_model_and_tokenizer("model", "cpu")

    assert first is second
    assert first[2] == 32
    assert tokenizer_loads == 1
    assert model_loads == 1
    assert model.loaded_device == "cpu"
    assert model.is_eval
    preparation._load_model_and_tokenizer.cache_clear()
