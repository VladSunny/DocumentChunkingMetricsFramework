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
