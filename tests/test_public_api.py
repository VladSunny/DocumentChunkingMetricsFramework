import importlib.util

import chunking_metrics
from chunking_metrics.metrics import size_compliance
from chunking_metrics.preparations import api, local
from chunking_metrics.preparations.api import calculation as api_calculation
from chunking_metrics.preparations.api import generation as api_generation
from chunking_metrics.preparations.local import calculate_embeddings
from chunking_metrics.preparations.local import calculation as local_calculation
from chunking_metrics.preparations.local import generation as local_generation
from chunking_metrics.prompts import DEFAULT_QUESTION_PROMPT


def test_public_api_is_grouped_by_module() -> None:
    assert chunking_metrics.__all__ == ["metrics", "preparations", "prompts"]
    assert chunking_metrics.preparations.__all__ == ["local", "api"]
    assert chunking_metrics.metrics.size_compliance is size_compliance
    assert chunking_metrics.preparations.local is local
    assert chunking_metrics.preparations.api is api
    assert local.calculate_embeddings is calculate_embeddings
    assert local.calculation is local_calculation
    assert local.generation is local_generation
    assert local.calculate_embeddings is local_calculation.calculate_embeddings
    assert local.calculate_perplexity is local_calculation.calculate_perplexity
    assert local.retrieve_relevant_chunks is local_calculation.retrieve_relevant_chunks
    assert local.generate_answers is local_generation.generate_answers
    assert local.generate_statements is local_generation.generate_statements
    assert local.generate_information_preservation_statements is (
        local_generation.generate_information_preservation_statements
    )
    assert local.evaluate_information_preservation is (
        local_generation.evaluate_information_preservation
    )
    assert local.generate_questions is local_generation.generate_questions
    assert api.calculation is api_calculation
    assert api.generation is api_generation
    assert api.calculate_embeddings is api_calculation.calculate_embeddings
    assert api.calculate_perplexity is api_calculation.calculate_perplexity
    assert api.generate_answers is api_generation.generate_answers
    assert api.generate_statements is api_generation.generate_statements
    assert api.generate_information_preservation_statements is (
        api_generation.generate_information_preservation_statements
    )
    assert api.evaluate_information_preservation is (
        api_generation.evaluate_information_preservation
    )
    assert api.generate_questions is api_generation.generate_questions
    assert chunking_metrics.prompts.DEFAULT_QUESTION_PROMPT is DEFAULT_QUESTION_PROMPT
    assert not hasattr(chunking_metrics, "size_compliance")
    assert not hasattr(chunking_metrics, "calculate_embeddings")
    assert not hasattr(chunking_metrics, "DEFAULT_QUESTION_PROMPT")
    assert importlib.util.find_spec("chunking_metrics.preparation") is None
