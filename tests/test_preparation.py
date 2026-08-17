import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import chunking_metrics.prompts as prompts
from chunking_metrics.preparations import api, local
from chunking_metrics.preparations.local import calculate_perplexity


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


def _api_embedding_response(
    items: list[tuple[int, list[float]]],
) -> SimpleNamespace:
    return SimpleNamespace(
        data=[
            SimpleNamespace(object="embedding", index=index, embedding=embedding)
            for index, embedding in items
        ],
        model="provider/embedding-model",
        object="list",
        usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
    )


def _api_perplexity_response(
    text_offsets: object,
    token_logprobs: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    text_offset=text_offsets,
                    token_logprobs=token_logprobs,
                )
            )
        ]
    )


def test_api_calculate_perplexity_returns_unconditional_score_and_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _api_perplexity_response(
                [0, 0, 3, 7],
                [None, -1.0, -3.0, -99.0],
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_arguments.append(kwargs)
            self.completions = FakeCompletions()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.calculate_perplexity(
        "  Привет",
        model_name=" provider/qwen ",
        api_key="secret",
        base_url="http://localhost:8000/v1",
    )

    assert result == pytest.approx(math.exp(2.0))
    assert client_arguments == [{"api_key": "secret", "base_url": "http://localhost:8000/v1"}]
    assert calls == [
        {
            "model": "provider/qwen",
            "prompt": " Привет",
            "max_tokens": 1,
            "echo": True,
            "logprobs": 0,
            "stream": False,
        }
    ]


def test_api_calculate_perplexity_scores_only_normalized_conditional_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _api_perplexity_response(
                [0, 0, 8, 9, 13],
                [None, -50.0, -math.log(2.0), -math.log(8.0), -100.0],
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.completions = FakeCompletions()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.calculate_perplexity(
        "  цель",
        "model",
        "secret",
        context="Контекст \t",
    )

    assert result == pytest.approx(4.0)
    assert calls[0]["prompt"] == "Контекст цель"


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("text", 1, TypeError, "text must be a string"),
        ("text", " \t", ValueError, "text must not be empty"),
        ("model_name", 1, TypeError, "model_name must be a string"),
        ("model_name", " ", ValueError, "model_name must not be empty"),
        ("api_key", 1, TypeError, "api_key must be a string"),
        ("api_key", " ", ValueError, "api_key must not be empty"),
        ("base_url", 1, TypeError, "base_url must be a string or None"),
        ("base_url", " ", ValueError, "base_url must not be empty"),
        ("context", 1, TypeError, "context must be a string or None"),
        ("context", " \t", ValueError, "context must not be empty"),
    ],
)
def test_api_calculate_perplexity_rejects_invalid_arguments_before_client(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    arguments: dict[str, object] = {
        "text": "text",
        "model_name": "model",
        "api_key": "secret",
        argument: value,
    }

    with pytest.raises(error_type, match=message):
        api.calculate_perplexity(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace()]),
        SimpleNamespace(choices=[SimpleNamespace(logprobs=None)]),
        SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(token_logprobs=[-1.0]))]),
        SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(text_offset=[0]))]),
        _api_perplexity_response([0], [-1.0, -2.0]),
        _api_perplexity_response([-1, 0, 5], [-1.0, -2.0, -3.0]),
        _api_perplexity_response([0, 2, 1, 5], [-1.0, -2.0, -3.0, -4.0]),
        _api_perplexity_response([0, 6], [-1.0, -2.0]),
        _api_perplexity_response([0, 1.5, 5], [-1.0, -2.0, -3.0]),
        _api_perplexity_response([0, 1, 5], [-1.0, None, -3.0]),
        _api_perplexity_response([0, 1, 5], [-1.0, float("nan"), -3.0]),
        _api_perplexity_response([0, 1, 5], [-1.0, float("inf"), -3.0]),
        _api_perplexity_response([0, 1, 5], [-1.0, "bad", -3.0]),
        _api_perplexity_response([0, 5], [None, -2.0]),
    ],
)
def test_api_calculate_perplexity_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    response: SimpleNamespace,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.completions = FakeCompletions()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(ValueError, match="API response"):
        api.calculate_perplexity("text", "model", "secret")


def test_api_calculate_perplexity_propagates_provider_errors_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_error = RuntimeError("context window exceeded")
    call_count = 0

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            raise provider_error

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.completions = FakeCompletions()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(RuntimeError) as raised:
        api.calculate_perplexity("text", "model", "secret")

    assert raised.value is provider_error
    assert call_count == 1


def test_api_calculate_embeddings_returns_single_float32_vector_and_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return _api_embedding_response([(0, [1.5, -2.0])])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_arguments.append(kwargs)
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.calculate_embeddings(
        " Text. ",
        model_name=" provider/embedding-model ",
        api_key="secret",
        base_url="https://example.test/v1",
    )

    np.testing.assert_array_equal(result, np.array([1.5, -2.0], dtype=np.float32))
    assert result.shape == (2,)
    assert result.dtype == np.float32
    assert client_arguments == [{"api_key": "secret", "base_url": "https://example.test/v1"}]
    assert calls == [
        {
            "model": "provider/embedding-model",
            "input": [" Text. "],
            "encoding_format": "float",
        }
    ]


def test_api_calculate_embeddings_batches_and_restores_global_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    responses = iter(
        [
            _api_embedding_response([(1, [2.0, 20.0]), (0, [1.0, 10.0])]),
            _api_embedding_response([(0, [3.0, 30.0])]),
        ]
    )

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return next(responses)

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.calculate_embeddings(
        ["first", "second", "third"],
        model_name="model",
        api_key="secret",
        dimensions=2,
        batch_size=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32),
    )
    assert calls == [
        {
            "model": "model",
            "input": ["first", "second"],
            "encoding_format": "float",
            "dimensions": 2,
        },
        {
            "model": "model",
            "input": ["third"],
            "encoding_format": "float",
            "dimensions": 2,
        },
    ]


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("texts", 1, TypeError, "texts must be a string or a sequence of strings"),
        ("texts", ["valid", 1], TypeError, r"texts\[1\] must be a string"),
        ("texts", [], ValueError, "texts must not be empty"),
        ("texts", ["  "], ValueError, r"texts\[0\] must not be empty"),
        ("model_name", 1, TypeError, "model_name must be a string"),
        ("model_name", "  ", ValueError, "model_name must not be empty"),
        ("api_key", 1, TypeError, "api_key must be a string"),
        ("api_key", "  ", ValueError, "api_key must not be empty"),
        ("base_url", 1, TypeError, "base_url must be a string or None"),
        ("base_url", "  ", ValueError, "base_url must not be empty"),
        ("dimensions", 1.5, TypeError, "dimensions must be an integer or None"),
        ("dimensions", True, TypeError, "dimensions must be an integer or None"),
        ("dimensions", 0, ValueError, "dimensions must be greater than zero"),
        ("batch_size", 1.5, TypeError, "batch_size must be an integer"),
        ("batch_size", True, TypeError, "batch_size must be an integer"),
        ("batch_size", 0, ValueError, "batch_size must be between 1 and 2048"),
        ("batch_size", 2049, ValueError, "batch_size must be between 1 and 2048"),
    ],
)
def test_api_calculate_embeddings_rejects_invalid_arguments_before_client(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    arguments: dict[str, object] = {
        "texts": ["first", "second"],
        "model_name": "model",
        "api_key": "secret",
        argument: value,
    }

    with pytest.raises(error_type, match=message):
        api.calculate_embeddings(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([(0, [1.0, 2.0])], "one embedding per input"),
        ([(0, [1.0, 2.0]), (0, [3.0, 4.0])], "unique embedding index"),
        ([(0, [1.0, 2.0]), (2, [3.0, 4.0])], "embedding indices from 0 to 1"),
        ([(0, [1.0, 2.0]), (1, [3.0])], "same non-zero dimension"),
        ([(0, []), (1, [])], "same non-zero dimension"),
        ([(0, [1.0, float("nan")]), (1, [3.0, 4.0])], "finite numbers"),
        ([(0, [1.0, float("inf")]), (1, [3.0, 4.0])], "finite numbers"),
    ],
)
def test_api_calculate_embeddings_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
    items: list[tuple[int, list[float]]],
    message: str,
) -> None:
    class FakeEmbeddings:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return _api_embedding_response(items)

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(ValueError, match=message):
        api.calculate_embeddings(["first", "second"], "model", "secret")


def test_api_calculate_embeddings_rejects_dimension_change_between_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _api_embedding_response([(0, [1.0, 2.0])]),
            _api_embedding_response([(0, [3.0, 4.0, 5.0])]),
        ]
    )

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return next(responses)

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(ValueError, match="same dimension across batches"):
        api.calculate_embeddings(["first", "second"], "model", "secret", batch_size=1)


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


class FakeAnswerTokenizer:
    eos_token_id = 2
    pad_token_id = 3

    def __init__(self, responses: list[str], prompt_lengths: list[int] | None = None) -> None:
        self.responses = responses
        self.prompt_lengths = prompt_lengths or [2] * len(responses)
        self.messages: list[list[dict[str, str]]] = []
        self.decode_index = 0

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
        prompt_length = self.prompt_lengths[len(self.messages)]
        self.messages.append(messages)
        return {
            "input_ids": torch.tensor([list(range(prompt_length))], dtype=torch.long),
            "attention_mask": torch.ones((1, prompt_length), dtype=torch.long),
        }

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        response = self.responses[self.decode_index]
        self.decode_index += 1
        return response


class FakeAnswerModel:
    def __init__(self) -> None:
        self.generation_arguments: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generation_arguments.append(kwargs)
        input_ids = kwargs["input_ids"]
        return torch.cat((input_ids, torch.tensor([[20]], dtype=torch.long)), dim=1)


@pytest.mark.parametrize(
    ("temperature", "expected_generation_arguments"),
    [
        (0.0, {"do_sample": False}),
        (0.4, {"do_sample": True, "temperature": 0.4}),
    ],
)
def test_generate_text_selects_decoding_mode_and_decodes_only_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
    temperature: float,
    expected_generation_arguments: dict[str, object],
) -> None:
    tokenizer = FakeStatementTokenizer(" Ответ. ")
    model = FakeStatementModel()
    loads: list[tuple[str, str]] = []

    def load_model(model_name: str, device: str) -> tuple[Any, Any, int]:
        loads.append((model_name, device))
        return tokenizer, model, 32

    monkeypatch.setattr(local, "_load_model_and_tokenizer", load_model)

    response = local._generate_text(
        [{"role": "user", "content": "Вопрос?"}],
        " model ",
        temperature=temperature,
        max_new_tokens=4,
        device="cpu",
        chat_template_error="missing template",
        context_window_error="context overflow",
    )

    assert response == " Ответ. "
    assert loads == [("model", "cpu")]
    assert tokenizer.decoded_ids == [20, 21]
    assert model.generation_arguments is not None
    for name, value in expected_generation_arguments.items():
        assert model.generation_arguments[name] == value
    if temperature == 0:
        assert "temperature" not in model.generation_arguments


def test_generate_text_uses_eos_token_for_padding_when_pad_token_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer("Ответ.")
    tokenizer.pad_token_id = None
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    local._generate_text(
        [{"role": "user", "content": "Вопрос?"}],
        "model",
        temperature=0.0,
        max_new_tokens=4,
        device="cpu",
        chat_template_error="missing template",
        context_window_error="context overflow",
    )

    assert model.generation_arguments is not None
    assert model.generation_arguments["pad_token_id"] == tokenizer.eos_token_id


def test_generate_text_rejects_prompt_and_response_that_exceed_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer("Ответ.")
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 5),
    )

    with pytest.raises(ValueError, match="custom context overflow"):
        local._generate_text(
            [{"role": "user", "content": "Вопрос?"}],
            "model",
            temperature=0.0,
            max_new_tokens=4,
            device="cpu",
            chat_template_error="missing template",
            context_window_error="custom context overflow",
        )

    assert model.generation_arguments is None


def test_generate_text_reports_custom_error_when_chat_template_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer("Ответ.")
    model = FakeStatementModel()

    def reject_chat_template(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Cannot use chat template functions")

    tokenizer.apply_chat_template = reject_chat_template  # type: ignore[method-assign]
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match="custom missing template"):
        local._generate_text(
            [{"role": "user", "content": "Вопрос?"}],
            "model",
            temperature=0.0,
            max_new_tokens=4,
            device="cpu",
            chat_template_error="custom missing template",
            context_window_error="context overflow",
        )


def install_fake_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_length: int = 32,
) -> FakeModel:
    tokenizer = FakeTokenizer(model_max_length=max_length)
    model = FakeModel(max_position_embeddings=max_length)
    monkeypatch.setattr(
        local,
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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_statements(
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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_statements(
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
        local,
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
        local.generate_statements(**arguments)  # type: ignore[arg-type]


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
        local,
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
        local.generate_statements(**call_arguments)  # type: ignore[arg-type]


def test_generate_statements_rejects_chunk_that_cannot_fit_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 9),
    )

    with pytest.raises(ValueError, match="do not fit within the model context window"):
        local.generate_statements(
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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match="tokenizer must define a chat template"):
        local.generate_statements("Текст чанка.", model_name="model", device="cpu")


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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(
        ValueError,
        match="model response must be a JSON array of exactly 2 non-empty strings",
    ):
        local.generate_statements(
            "Текст чанка.",
            model_name="model",
            statement_count=2,
            max_new_tokens=8,
            device="cpu",
        )


def test_generate_statements_is_available_from_local_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первое.", "Второе."]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_statements(
        "Текст чанка.",
        model_name="model",
        statement_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первое.", "Второе."]
    assert local.DEFAULT_STATEMENT_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"


def test_generate_information_preservation_statements_returns_labeled_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"true_statement":" True. ",'
                                '"false_statements":[" False 1. ","False 2.","False 3."]}'
                            )
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_arguments.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)
    generate = getattr(api, "generate_information_preservation_statements", None)

    assert callable(generate)
    result = generate(
        "Three sentence segment.",
        model_name="provider/model",
        api_key="secret",
        base_url="https://example.test/v1",
        temperature=0.2,
        max_new_tokens=128,
    )

    assert result == ("True.", ["False 1.", "False 2.", "False 3."])
    assert client_arguments == [{"api_key": "secret", "base_url": "https://example.test/v1"}]
    assert len(calls) == 1
    assert calls[0]["model"] == "provider/model"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 128
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["messages"][0] == {  # type: ignore[index]
        "role": "system",
        "content": prompts.DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
    }
    assert (
        '<source>\n"Three sentence segment."\n</source>'
        in calls[0]["messages"][1][  # type: ignore[index]
            "content"
        ]
    )


def test_local_generate_information_preservation_statements_uses_local_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def generate_text(messages: list[dict[str, str]], model_name: str, **kwargs: object) -> str:
        calls.append({"messages": messages, "model_name": model_name, **kwargs})
        return (
            '{"true_statement":" True. ","false_statements":[" False 1. ","False 2.","False 3."]}'
        )

    monkeypatch.setattr(local, "_generate_text", generate_text)

    result = local.generate_information_preservation_statements(
        "Three sentence segment.",
        prompt="Source: {segment}",
        temperature=0.2,
        max_new_tokens=128,
    )

    assert result == ("True.", ["False 1.", "False 2.", "False 3."])
    assert calls == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": prompts.DEFAULT_INFORMATION_PRESERVATION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": 'Source: "Three sentence segment."'},
            ],
            "model_name": local.DEFAULT_STATEMENT_MODEL,
            "temperature": 0.2,
            "max_new_tokens": 128,
            "device": None,
            "chat_template_error": "model tokenizer must define a chat template",
            "context_window_error": (
                "segment and generated response do not fit within the model context window"
            ),
        }
    ]


def test_local_evaluate_information_preservation_scores_and_preserves_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def generate_text(messages: list[dict[str, str]], model_name: str, **kwargs: object) -> str:
        calls.append({"messages": messages, "model_name": model_name, **kwargs})
        return '{"selected_index": 3}'

    false_statements = ["False 1.", "False 2.", "False 3."]
    relevant_chunks = ["Chunk one.", "Chunk two."]
    monkeypatch.setattr(local, "_generate_text", generate_text)

    result = local.evaluate_information_preservation(
        "True.",
        false_statements,
        relevant_chunks,
        prompt="Options: {statements}\nChunks: {relevant_chunks}",
        temperature=0.2,
        max_new_tokens=17,
        seed=7,
        device="cpu",
    )

    assert result == 1
    assert false_statements == ["False 1.", "False 2.", "False 3."]
    assert relevant_chunks == ["Chunk one.", "Chunk two."]
    assert calls == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": prompts.DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        'Options: [{"index": 1, "statement": "False 3."}, '
                        '{"index": 2, "statement": "False 1."}, '
                        '{"index": 3, "statement": "True."}, '
                        '{"index": 4, "statement": "False 2."}]\n'
                        'Chunks: ["Chunk one.", "Chunk two."]'
                    ),
                },
            ],
            "model_name": local.DEFAULT_STATEMENT_MODEL,
            "temperature": 0.2,
            "max_new_tokens": 17,
            "device": "cpu",
            "chat_template_error": "model tokenizer must define a chat template",
            "context_window_error": (
                "statements, relevant chunks, and generated response do not fit within the "
                "model context window"
            ),
        }
    ]


def test_local_evaluate_information_preservation_returns_zero_and_repeats_seeded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_messages: list[str] = []

    def generate_text(messages: list[dict[str, str]], *args: object, **kwargs: object) -> str:
        user_messages.append(messages[-1]["content"])
        return '{"selected_index": 1}'

    monkeypatch.setattr(local, "_generate_text", generate_text)
    arguments = {
        "true_statement": "True.",
        "false_statements": ["False 1.", "False 2.", "False 3."],
        "relevant_chunks": ["Relevant chunk."],
        "seed": 7,
    }

    first_result = local.evaluate_information_preservation(**arguments)
    second_result = local.evaluate_information_preservation(**arguments)

    assert first_result == second_result == 0
    assert user_messages[0] == user_messages[1]


@pytest.mark.parametrize(
    ("function_name", "arguments", "error_type", "message"),
    [
        (
            "generate_information_preservation_statements",
            {"segment": 1},
            TypeError,
            "segment must be a string",
        ),
        (
            "generate_information_preservation_statements",
            {"segment": "Segment.", "device": 1},
            TypeError,
            "device must be a string or None",
        ),
        (
            "generate_information_preservation_statements",
            {"segment": "  "},
            ValueError,
            "segment must not be empty",
        ),
        (
            "generate_information_preservation_statements",
            {"segment": "Segment.", "prompt": "No placeholder."},
            ValueError,
            r"prompt must contain \{segment\}",
        ),
        (
            "generate_information_preservation_statements",
            {"segment": "Segment.", "temperature": 0},
            ValueError,
            "temperature must be greater than zero",
        ),
        (
            "generate_information_preservation_statements",
            {"segment": "Segment.", "max_new_tokens": 0},
            ValueError,
            "max_new_tokens must be greater than zero",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F1.", "F3."],
                "relevant_chunks": ["Chunk."],
            },
            ValueError,
            "false_statements must be distinct",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": [],
            },
            ValueError,
            "relevant_chunks must not be empty",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": ["Chunk."],
                "prompt": "Only {statements}.",
            },
            ValueError,
            r"prompt must contain \{statements\} and \{relevant_chunks\}",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": ["Chunk."],
                "temperature": -0.1,
            },
            ValueError,
            "temperature must be greater than or equal to zero",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": ["Chunk."],
                "device": 1,
            },
            TypeError,
            "device must be a string or None",
        ),
    ],
)
def test_local_information_preservation_rejects_invalid_arguments_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    arguments: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        local,
        "_generate_text",
        lambda *args, **kwargs: pytest.fail("generation must not start"),
    )

    with pytest.raises(error_type, match=message):
        getattr(local, function_name)(**arguments)


@pytest.mark.parametrize(
    ("function_name", "arguments", "response", "message"),
    [
        (
            "generate_information_preservation_statements",
            {"segment": "Segment."},
            "not JSON",
            "model response must be a JSON object with one non-empty true_statement",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": ["Chunk."],
            },
            '{"selected_index": 5}',
            "model response must be a JSON object containing only selected_index from 1 to 4",
        ),
    ],
)
def test_local_information_preservation_rejects_invalid_json_response(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    arguments: dict[str, object],
    response: str,
    message: str,
) -> None:
    monkeypatch.setattr(local, "_generate_text", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match=message):
        getattr(local, function_name)(**arguments)


@pytest.mark.parametrize(
    ("function_name", "arguments", "template_error", "context_error"),
    [
        (
            "generate_information_preservation_statements",
            {"segment": "Segment."},
            "model tokenizer must define a chat template",
            "segment and generated response do not fit within the model context window",
        ),
        (
            "evaluate_information_preservation",
            {
                "true_statement": "True.",
                "false_statements": ["F1.", "F2.", "F3."],
                "relevant_chunks": ["Chunk."],
            },
            "model tokenizer must define a chat template",
            (
                "statements, relevant chunks, and generated response do not fit within the "
                "model context window"
            ),
        ),
    ],
)
def test_local_information_preservation_reports_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    arguments: dict[str, object],
    template_error: str,
    context_error: str,
) -> None:
    tokenizer = FakeStatementTokenizer("unused")
    tokenizer.apply_chat_template = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing template"))
    )
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, FakeStatementModel(), 32),
    )
    with pytest.raises(ValueError, match=template_error):
        getattr(local, function_name)(**arguments)

    tokenizer = FakeStatementTokenizer("unused")
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, FakeStatementModel(), 5),
    )
    with pytest.raises(ValueError, match=context_error):
        getattr(local, function_name)(**arguments)


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        '{"true_statement":"True.","false_statements":["F1.","F2.","F3."],"extra":1}',
        '{"true_statement":"True."}',
        '{"true_statement":"  ","false_statements":["F1.","F2.","F3."]}',
        '{"true_statement":"True.","false_statements":["F1.","F2."]}',
        '{"true_statement":"True.","false_statements":["F1.",2,"F3."]}',
        '{"true_statement":"True.","false_statements":["F1.","F1.","F3."]}',
        '{"true_statement":"True.","false_statements":["F1.","True.","F3."]}',
        None,
    ],
)
def test_generate_information_preservation_statements_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(
        ValueError,
        match=(
            "model response must be a JSON object with one non-empty true_statement and "
            "exactly three distinct non-empty false_statements"
        ),
    ):
        api.generate_information_preservation_statements(
            "Segment.", model_name="model", api_key="secret"
        )


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("segment", 1, TypeError, "segment must be a string"),
        ("model_name", 1, TypeError, "model_name must be a string"),
        ("api_key", 1, TypeError, "api_key must be a string"),
        ("base_url", 1, TypeError, "base_url must be a string or None"),
        ("prompt", 1, TypeError, "prompt must be a string"),
        ("temperature", True, TypeError, "temperature must be a number"),
        ("max_new_tokens", True, TypeError, "max_new_tokens must be an integer"),
        ("segment", "  ", ValueError, "segment must not be empty"),
        ("model_name", "  ", ValueError, "model_name must not be empty"),
        ("api_key", "  ", ValueError, "api_key must not be empty"),
        ("base_url", "  ", ValueError, "base_url must not be empty"),
        ("prompt", "  ", ValueError, "prompt must not be empty"),
        ("prompt", "No placeholder.", ValueError, r"prompt must contain \{segment\}"),
        ("prompt", "{segment", ValueError, "prompt must be a valid format string"),
        ("prompt", "{segment:d}", ValueError, "prompt must be a valid format string"),
        (
            "prompt",
            "{segment} {unknown}",
            ValueError,
            "prompt contains an unsupported placeholder: unknown",
        ),
        ("temperature", 0, ValueError, "temperature must be greater than zero"),
        ("temperature", float("nan"), ValueError, "temperature must be finite"),
        ("max_new_tokens", 0, ValueError, "max_new_tokens must be greater than zero"),
    ],
)
def test_generate_information_preservation_statements_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    arguments: dict[str, object] = {
        "segment": "Segment.",
        "model_name": "model",
        "api_key": "secret",
        "base_url": None,
        argument: value,
    }

    with pytest.raises(error_type, match=message):
        api.generate_information_preservation_statements(  # type: ignore[arg-type]
            **arguments
        )


def test_evaluate_information_preservation_returns_one_and_sends_exact_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"selected_index": 3}'))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_arguments.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    true_statement = "True."
    false_statements = ["False 1.", "False 2.", "False 3."]
    relevant_chunks = ["Chunk one.", "Chunk two."]
    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.evaluate_information_preservation(
        true_statement,
        false_statements,
        relevant_chunks,
        model_name="provider/model",
        api_key="secret",
        base_url="https://example.test/v1",
        prompt="Options: {statements}\nChunks: {relevant_chunks}",
        temperature=0.2,
        max_new_tokens=17,
        seed=7,
    )

    assert result == 1
    assert false_statements == ["False 1.", "False 2.", "False 3."]
    assert relevant_chunks == ["Chunk one.", "Chunk two."]
    assert client_arguments == [{"api_key": "secret", "base_url": "https://example.test/v1"}]
    assert calls == [
        {
            "model": "provider/model",
            "messages": [
                {
                    "role": "system",
                    "content": prompts.DEFAULT_INFORMATION_PRESERVATION_EVALUATION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        'Options: [{"index": 1, "statement": "False 3."}, '
                        '{"index": 2, "statement": "False 1."}, '
                        '{"index": 3, "statement": "True."}, '
                        '{"index": 4, "statement": "False 2."}]\n'
                        'Chunks: ["Chunk one.", "Chunk two."]'
                    ),
                },
            ],
            "stream": False,
            "extra_body": {"thinking": {"type": "disabled"}},
            "response_format": {"type": "json_object"},
            "max_tokens": 17,
            "temperature": 0.2,
        }
    ]


def test_evaluate_information_preservation_returns_zero_for_false_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"selected_index": 1}'))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.evaluate_information_preservation(
        "True.",
        ["False 1.", "False 2.", "False 3."],
        ["Relevant chunk."],
        model_name="model",
        api_key="secret",
        seed=7,
    )

    assert result == 0


def test_evaluate_information_preservation_repeats_order_for_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_messages: list[str] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            user_messages.append(kwargs["messages"][-1]["content"])  # type: ignore[index]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"selected_index": 1}'))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)
    arguments = {
        "true_statement": "True.",
        "false_statements": ["False 1.", "False 2.", "False 3."],
        "relevant_chunks": ["Relevant chunk."],
        "model_name": "model",
        "api_key": "secret",
        "seed": 7,
    }

    api.evaluate_information_preservation(**arguments)
    api.evaluate_information_preservation(**arguments)

    assert user_messages[0] == user_messages[1]


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("true_statement", 1, "true_statement must be a string"),
        ("false_statements", "false", "false_statements must be a sequence of strings"),
        ("false_statements", ["F1.", 2, "F3."], r"false_statements\[1\] must be a string"),
        ("relevant_chunks", "chunk", "relevant_chunks must be a sequence of strings"),
        ("relevant_chunks", [1], r"relevant_chunks\[0\] must be a string"),
        ("model_name", 1, "model_name must be a string"),
        ("api_key", 1, "api_key must be a string"),
        ("base_url", 1, "base_url must be a string or None"),
        ("prompt", 1, "prompt must be a string"),
        ("temperature", True, "temperature must be a number"),
        ("max_new_tokens", True, "max_new_tokens must be an integer"),
        ("seed", True, "seed must be an integer or None"),
        ("seed", 1.5, "seed must be an integer or None"),
    ],
)
def test_evaluate_information_preservation_rejects_invalid_types_before_client(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    arguments: dict[str, object] = {
        "true_statement": "True.",
        "false_statements": ["F1.", "F2.", "F3."],
        "relevant_chunks": ["Chunk."],
        "model_name": "model",
        "api_key": "secret",
        argument: value,
    }

    with pytest.raises(TypeError, match=message):
        api.evaluate_information_preservation(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"true_statement": "  "}, "true_statement must not be empty"),
        ({"false_statements": []}, "false_statements must contain exactly three statements"),
        (
            {"false_statements": ["F1.", "F2."]},
            "false_statements must contain exactly three statements",
        ),
        (
            {"false_statements": ["F1.", "F2.", "F3.", "F4."]},
            "false_statements must contain exactly three statements",
        ),
        (
            {"false_statements": ["F1.", "  ", "F3."]},
            r"false_statements\[1\] must not be empty",
        ),
        (
            {"false_statements": ["F1.", "F1. ", "F3."]},
            "false_statements must be distinct",
        ),
        (
            {"false_statements": ["F1.", "True. ", "F3."]},
            "false_statements must not contain true_statement",
        ),
        ({"relevant_chunks": []}, "relevant_chunks must not be empty"),
        ({"relevant_chunks": ["  "]}, r"relevant_chunks\[0\] must not be empty"),
        ({"model_name": "  "}, "model_name must not be empty"),
        ({"api_key": "  "}, "api_key must not be empty"),
        ({"base_url": "  "}, "base_url must not be empty"),
        ({"prompt": "  "}, "prompt must not be empty"),
        (
            {"prompt": "Only {statements}."},
            r"prompt must contain \{statements\} and \{relevant_chunks\}",
        ),
        (
            {"prompt": "Only {relevant_chunks}."},
            r"prompt must contain \{statements\} and \{relevant_chunks\}",
        ),
        ({"prompt": "{statements"}, "prompt must be a valid format string"),
        (
            {"prompt": "{statements} {relevant_chunks} {unknown}"},
            "prompt contains an unsupported placeholder: unknown",
        ),
        ({"temperature": -0.1}, "temperature must be greater than or equal to zero"),
        ({"temperature": float("nan")}, "temperature must be finite"),
        ({"temperature": float("inf")}, "temperature must be finite"),
        ({"max_new_tokens": 0}, "max_new_tokens must be greater than zero"),
    ],
)
def test_evaluate_information_preservation_rejects_invalid_values_before_client(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    call_arguments: dict[str, object] = {
        "true_statement": "True.",
        "false_statements": ["F1.", "F2.", "F3."],
        "relevant_chunks": ["Chunk."],
        "model_name": "model",
        "api_key": "secret",
        **arguments,
    }

    with pytest.raises(ValueError, match=message):
        api.evaluate_information_preservation(  # type: ignore[arg-type]
            **call_arguments
        )


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        "{}",
        '{"selected_index": 1, "extra": 2}',
        '{"selected_index": true}',
        '{"selected_index": 1.5}',
        '{"selected_index": 0}',
        '{"selected_index": 5}',
        None,
    ],
)
def test_evaluate_information_preservation_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(
        ValueError,
        match=r"model response must be a JSON object containing only selected_index from 1 to 4",
    ):
        api.evaluate_information_preservation(
            "True.",
            ["F1.", "F2.", "F3."],
            ["Chunk."],
            model_name="model",
            api_key="secret",
        )


def test_generate_questions_returns_exact_cleaned_json_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('[" Первый вопрос? ", "Второй вопрос?"]')
    model = FakeStatementModel()

    def load_model(model_name: str, device: str) -> tuple[Any, Any, int]:
        assert model_name == local.DEFAULT_STATEMENT_MODEL
        assert device == "cpu"
        return tokenizer, model, 32

    monkeypatch.setattr(local, "_load_model_and_tokenizer", load_model)

    result = local.generate_questions(
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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_questions(
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
        local,
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
        local.generate_questions(**arguments)  # type: ignore[arg-type]


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
        local,
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
        local.generate_questions(**call_arguments)  # type: ignore[arg-type]


def test_generate_questions_rejects_chunk_that_cannot_fit_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 9),
    )

    with pytest.raises(ValueError, match="do not fit within the model context window"):
        local.generate_questions(
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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match="tokenizer must define a chat template"):
        local.generate_questions("Текст чанка.", model_name="model", device="cpu")


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
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(
        ValueError,
        match="model response must be a JSON array of exactly 2 non-empty strings",
    ):
        local.generate_questions(
            "Текст чанка.",
            model_name="model",
            question_count=2,
            max_new_tokens=8,
            device="cpu",
        )


def test_generate_questions_is_available_from_local_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeStatementTokenizer('["Первый?", "Второй?"]')
    model = FakeStatementModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_questions(
        "Текст чанка.",
        model_name="model",
        question_count=2,
        max_new_tokens=8,
        device="cpu",
    )

    assert result == ["Первый?", "Второй?"]


def test_generate_answers_returns_answers_in_order_with_per_question_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeAnswerTokenizer([" Первый ответ. ", "Второй ответ."])
    model = FakeAnswerModel()
    loads: list[tuple[str, str]] = []

    def load_model(model_name: str, device: str) -> tuple[Any, Any, int]:
        loads.append((model_name, device))
        return tokenizer, model, 32

    monkeypatch.setattr(local, "_load_model_and_tokenizer", load_model)

    result = local.generate_answers(
        ["Первый вопрос?", "Второй вопрос?"],
        "Основной чанк.",
        model_name="model",
        additional_chunks_by_question=[["Дополнение 1."], ["Дополнение 2.", "Дополнение 3."]],
        max_new_tokens=4,
        device="cpu",
    )

    assert result == ["Первый ответ.", "Второй ответ."]
    assert loads
    assert set(loads) == {("model", "cpu")}
    assert len(model.generation_arguments) == 2
    first_prompt = tokenizer.messages[0][-1]["content"]
    second_prompt = tokenizer.messages[1][-1]["content"]
    assert '"Первый вопрос?"' in first_prompt
    assert '<primary_source>\n"Основной чанк."\n</primary_source>' in first_prompt
    assert '<additional_sources>\n["Дополнение 1."]\n</additional_sources>' in first_prompt
    assert '"Второй вопрос?"' in second_prompt
    assert (
        '<additional_sources>\n["Дополнение 2.", "Дополнение 3."]\n</additional_sources>'
        in second_prompt
    )


@pytest.mark.parametrize(
    ("temperature", "expected_generation_arguments"),
    [
        (0.0, {"do_sample": False}),
        (0.4, {"do_sample": True, "temperature": 0.4}),
    ],
)
def test_generate_answers_selects_greedy_or_sampling_decoding(
    monkeypatch: pytest.MonkeyPatch,
    temperature: float,
    expected_generation_arguments: dict[str, object],
) -> None:
    tokenizer = FakeAnswerTokenizer(["Ответ."])
    model = FakeAnswerModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    result = local.generate_answers(
        ["Вопрос?"],
        "Чанк.",
        model_name="model",
        temperature=temperature,
        max_new_tokens=4,
        device="cpu",
    )

    assert result == ["Ответ."]
    generation_arguments = model.generation_arguments[0]
    for name, value in expected_generation_arguments.items():
        assert generation_arguments[name] == value
    if temperature == 0:
        assert "temperature" not in generation_arguments
    assert "<additional_sources>\n[]\n</additional_sources>" in tokenizer.messages[0][-1]["content"]


def test_generate_answers_creates_one_client_and_calls_once_per_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_arguments: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    contents = iter(["Первый API-ответ.", " Второй API-ответ. "])

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=next(contents)))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            client_arguments.append(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    result = api.generate_answers(
        ["Первый?", "Второй?"],
        "Чанк.",
        model_name="provider/model",
        api_key="secret",
        base_url="https://example.test/v1",
        additional_chunks_by_question=[[], ["Контекст."]],
        temperature=0.2,
        max_new_tokens=17,
    )

    assert result == ["Первый API-ответ.", "Второй API-ответ."]
    assert client_arguments == [{"api_key": "secret", "base_url": "https://example.test/v1"}]
    assert len(calls) == 2
    assert all(
        {name: call[name] for name in ("model", "temperature", "max_tokens")}
        == {"model": "provider/model", "temperature": 0.2, "max_tokens": 17}
        for call in calls
    )
    assert all("response_format" not in call and "extra_body" not in call for call in calls)
    assert '"Первый?"' in calls[0]["messages"][-1]["content"]  # type: ignore[index]
    assert '["Контекст."]' in calls[1]["messages"][-1]["content"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("questions", "not-a-sequence", "questions must be a sequence of strings"),
        ("questions", ["Вопрос?", 1], r"questions\[1\] must be a string"),
        ("chunk", 1, "chunk must be a string"),
        ("model_name", 1, "model_name must be a string"),
        ("prompt", 1, "prompt must be a string"),
        ("temperature", True, "temperature must be a number"),
        ("max_new_tokens", True, "max_new_tokens must be an integer"),
        ("device", 1, "device must be a string or None"),
        (
            "additional_chunks_by_question",
            "context",
            "additional_chunks_by_question must be a sequence of string sequences or None",
        ),
        (
            "additional_chunks_by_question",
            [["Контекст."], "context"],
            r"additional_chunks_by_question\[1\] must be a sequence of strings",
        ),
        (
            "additional_chunks_by_question",
            [["Контекст."], [1]],
            r"additional_chunks_by_question\[1\]\[0\] must be a string",
        ),
    ],
)
def test_generate_answers_rejects_invalid_argument_types_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    arguments: dict[str, object] = {
        "questions": ["Первый?", "Второй?"],
        "chunk": "Чанк.",
        "model_name": "model",
        "max_new_tokens": 4,
        "device": "cpu",
        argument: value,
    }

    with pytest.raises(TypeError, match=message):
        local.generate_answers(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"questions": []}, "questions must not be empty"),
        ({"questions": ["  "]}, r"questions\[0\] must not be empty"),
        ({"chunk": "  "}, "chunk must not be empty"),
        ({"model_name": "  "}, "model_name must not be empty"),
        ({"additional_chunks_by_question": [[]]}, "must contain one item per question"),
        (
            {"additional_chunks_by_question": [["Контекст."], ["  "]]},
            r"additional_chunks_by_question\[1\]\[0\] must not be empty",
        ),
        (
            {"prompt": "Question: {question}; chunk: {chunk}"},
            r"prompt must contain \{question\}, \{chunk\}, and \{additional_chunks\}",
        ),
        ({"prompt": "{question} {chunk} {additional_chunks"}, "valid format string"),
        (
            {"prompt": ("{question} {chunk} {additional_chunks} {unsupported}")},
            "prompt contains an unsupported placeholder: unsupported",
        ),
        ({"temperature": -0.1}, "temperature must be greater than or equal to zero"),
        ({"temperature": float("nan")}, "temperature must be finite"),
        ({"max_new_tokens": 0}, "max_new_tokens must be greater than zero"),
    ],
)
def test_generate_answers_rejects_invalid_argument_values_before_loading_model(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )
    call_arguments: dict[str, object] = {
        "questions": ["Первый?", "Второй?"],
        "chunk": "Чанк.",
        "model_name": "model",
        "additional_chunks_by_question": [[], []],
        "max_new_tokens": 4,
        "device": "cpu",
        **arguments,
    }

    with pytest.raises(ValueError, match=message):
        local.generate_answers(**call_arguments)  # type: ignore[arg-type]


def test_generate_answers_reports_question_index_on_context_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeAnswerTokenizer(["Первый."], prompt_lengths=[2, 7])
    model = FakeAnswerModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 10),
    )

    with pytest.raises(ValueError, match=r"question 1.*model context window"):
        local.generate_answers(
            ["Первый?", "Второй?"],
            "Чанк.",
            model_name="model",
            max_new_tokens=4,
            device="cpu",
        )

    assert len(model.generation_arguments) == 1


def test_generate_answers_requires_tokenizer_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeAnswerTokenizer(["Ответ."])
    model = FakeAnswerModel()

    def reject_chat_template(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Cannot use chat template functions")

    tokenizer.apply_chat_template = reject_chat_template  # type: ignore[method-assign]
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match=r"question 0.*tokenizer must define a chat template"):
        local.generate_answers(
            ["Вопрос?"], "Чанк.", model_name="model", max_new_tokens=4, device="cpu"
        )


@pytest.mark.parametrize("response", ["", "  "])
def test_generate_answers_rejects_empty_answer(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    tokenizer = FakeAnswerTokenizer([response])
    model = FakeAnswerModel()
    monkeypatch.setattr(
        local,
        "_load_model_and_tokenizer",
        lambda model_name, device: (tokenizer, model, 32),
    )

    with pytest.raises(ValueError, match=r"answer for question 0 must not be empty"):
        local.generate_answers(
            ["Вопрос?"], "Чанк.", model_name="model", max_new_tokens=4, device="cpu"
        )


def test_generate_answers_rejects_empty_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(api, "OpenAI", FakeOpenAI)

    with pytest.raises(ValueError, match=r"answer for question 0 must not be empty"):
        api.generate_answers(["Вопрос?"], "Чанк.", model_name="model", api_key="key")


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("api_key", 1, TypeError, "api_key must be a string"),
        ("base_url", 1, TypeError, "base_url must be a string or None"),
        ("api_key", "  ", ValueError, "api_key must not be empty"),
        ("base_url", "  ", ValueError, "base_url must not be empty"),
    ],
)
def test_generate_answers_rejects_invalid_client_arguments_before_creating_client(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        api,
        "OpenAI",
        lambda **kwargs: pytest.fail("client must not be created"),
    )
    arguments: dict[str, object] = {
        "questions": ["Вопрос?"],
        "chunk": "Чанк.",
        "model_name": "model",
        "api_key": "key",
        "base_url": None,
        argument: value,
    }

    with pytest.raises(error_type, match=message):
        api.generate_answers(**arguments)  # type: ignore[arg-type]


def test_calculate_embeddings_returns_vector_for_single_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(
        local,
        "_load_embedding_model",
        lambda model_name, device: model,
        raising=False,
    )

    result = local.calculate_embeddings(
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
        local,
        "_load_embedding_model",
        lambda model_name, device: model,
        raising=False,
    )

    result = local.calculate_embeddings(
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

    monkeypatch.setattr(local, "SentenceTransformer", load_model, raising=False)
    local._load_embedding_model.cache_clear()

    first = local._load_embedding_model("model", "cpu")
    second = local._load_embedding_model("model", "cpu")

    assert first is second
    assert loads == [("model", "cpu")]
    assert model.is_eval
    local._load_embedding_model.cache_clear()


def test_calculate_embeddings_rejects_empty_text_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local,
        "_load_embedding_model",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )

    with pytest.raises(ValueError, match="texts must not be empty"):
        local.calculate_embeddings([], device="cpu")


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
        local,
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
        local.calculate_embeddings(**arguments)  # type: ignore[arg-type]


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
        local,
        "_load_embedding_model",
        lambda model_name, device: pytest.fail("model must not be loaded"),
    )

    with pytest.raises(ValueError, match=message):
        local.calculate_embeddings(**arguments, device="cpu")  # type: ignore[arg-type]


def test_calculate_embeddings_warns_when_texts_are_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel(max_seq_length=5)
    monkeypatch.setattr(
        local,
        "_load_embedding_model",
        lambda model_name, device: model,
    )

    with pytest.warns(
        UserWarning,
        match="2 texts exceed the model limit of 5 tokens and will be truncated",
    ):
        result = local.calculate_embeddings(
            ["abcd", "x", "abcdef"],
            model_name="model",
            device="cpu",
            batch_size=4,
        )

    assert result.shape == (3, 2)


def test_calculate_embeddings_is_available_from_local_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeEmbeddingModel()
    monkeypatch.setattr(
        local,
        "_load_embedding_model",
        lambda model_name, device: model,
    )

    result = local.calculate_embeddings(
        "text",
        model_name="model",
        device="cpu",
        batch_size=4,
    )

    assert result.shape == (2,)


def test_retrieve_relevant_chunks_ranks_candidates_for_each_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str, str | None, int]] = []

    def calculate_embeddings(
        texts: list[str],
        model_name: str,
        *,
        device: str | None,
        batch_size: int,
    ) -> np.ndarray:
        calls.append((texts, model_name, device, batch_size))
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(local, "calculate_embeddings", calculate_embeddings)

    result = local.retrieve_relevant_chunks(
        ["query-a", "query-b"],
        ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
        model_name="model",
        device="cpu",
        batch_size=4,
    )

    assert result == [
        ["chunk-c", "chunk-a", "chunk-b"],
        ["chunk-b", "chunk-a", "chunk-c"],
    ]
    assert calls == [
        (
            ["query-a", "query-b", "chunk-a", "chunk-b", "chunk-c", "chunk-d"],
            "model",
            "cpu",
            4,
        )
    ]


def test_retrieve_relevant_chunks_returns_all_candidates_and_preserves_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local,
        "calculate_embeddings",
        lambda *args, **kwargs: np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )

    result = local.retrieve_relevant_chunks(
        ["query"],
        ["first", "second"],
        top_k=3,
    )

    assert result == [["first", "second"]]


@pytest.mark.parametrize(
    ("argument", "value", "error_type", "message"),
    [
        ("queries", "query", TypeError, "queries must be a sequence of strings"),
        ("queries", ["query", 1], TypeError, r"queries\[1\] must be a string"),
        (
            "candidate_chunks",
            "chunk",
            TypeError,
            "candidate_chunks must be a sequence of strings",
        ),
        (
            "candidate_chunks",
            ["chunk", 1],
            TypeError,
            r"candidate_chunks\[1\] must be a string",
        ),
        ("top_k", 1.5, TypeError, "top_k must be an integer"),
        ("top_k", True, TypeError, "top_k must be an integer"),
    ],
)
def test_retrieve_relevant_chunks_rejects_invalid_argument_types_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        local,
        "calculate_embeddings",
        lambda *args, **kwargs: pytest.fail("embeddings must not be calculated"),
    )
    arguments: dict[str, object] = {
        "queries": ["query"],
        "candidate_chunks": ["chunk"],
        argument: value,
    }

    with pytest.raises(error_type, match=message):
        local.retrieve_relevant_chunks(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("queries", [], "queries must not be empty"),
        ("queries", ["query", "  "], r"queries\[1\] must not be empty"),
        ("candidate_chunks", [], "candidate_chunks must not be empty"),
        (
            "candidate_chunks",
            ["chunk", "\t"],
            r"candidate_chunks\[1\] must not be empty",
        ),
        ("top_k", 0, "top_k must be greater than zero"),
        ("top_k", -1, "top_k must be greater than zero"),
    ],
)
def test_retrieve_relevant_chunks_rejects_invalid_values_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        local,
        "calculate_embeddings",
        lambda *args, **kwargs: pytest.fail("embeddings must not be calculated"),
    )
    arguments: dict[str, object] = {
        "queries": ["query"],
        "candidate_chunks": ["chunk"],
        argument: value,
    }

    with pytest.raises(ValueError, match=message):
        local.retrieve_relevant_chunks(**arguments)  # type: ignore[arg-type]


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

    assert local._resolve_device(None) == "cuda"


def test_resolve_device_uses_mps_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert local._resolve_device(None) == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert local._resolve_device(None) == "cpu"


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

    monkeypatch.setattr(local.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(local.AutoModelForCausalLM, "from_pretrained", load_model)
    local._load_model_and_tokenizer.cache_clear()

    first = local._load_model_and_tokenizer("model", "cpu")
    second = local._load_model_and_tokenizer("model", "cpu")

    assert first is second
    assert first[2] == 32
    assert tokenizer_loads == 1
    assert model_loads == 1
    assert model.loaded_device == "cpu"
    assert model.is_eval
    local._load_model_and_tokenizer.cache_clear()
