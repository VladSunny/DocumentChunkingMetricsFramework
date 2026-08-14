import importlib.util

import chunking_metrics
from chunking_metrics.metrics import size_compliance
from chunking_metrics.preparations import api, local
from chunking_metrics.preparations.local import calculate_embeddings
from chunking_metrics.prompts import DEFAULT_QUESTION_PROMPT


def test_public_api_is_grouped_by_module() -> None:
    assert chunking_metrics.__all__ == ["metrics", "preparations", "prompts"]
    assert chunking_metrics.preparations.__all__ == ["local", "api"]
    assert chunking_metrics.metrics.size_compliance is size_compliance
    assert chunking_metrics.preparations.local is local
    assert chunking_metrics.preparations.api is api
    assert local.calculate_embeddings is calculate_embeddings
    assert chunking_metrics.prompts.DEFAULT_QUESTION_PROMPT is DEFAULT_QUESTION_PROMPT
    assert not hasattr(chunking_metrics, "size_compliance")
    assert not hasattr(chunking_metrics, "calculate_embeddings")
    assert not hasattr(chunking_metrics, "DEFAULT_QUESTION_PROMPT")
    assert importlib.util.find_spec("chunking_metrics.preparation") is None
