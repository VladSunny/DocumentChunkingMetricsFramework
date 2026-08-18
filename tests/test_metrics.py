import numpy as np
import pytest

import chunking_metrics.metrics as metrics
from chunking_metrics.metrics import (
    boundary_clarity,
    chunk_score,
    concept_unity,
    contextual_coherence,
    coreference_integrity,
    intrachunk_cohesion,
    semantic_dispersion,
    size_compliance,
)


def test_size_compliance_returns_score_for_each_length_in_input_order() -> None:
    lengths = np.array([5, 10, 20])
    result = size_compliance(
        lengths=lengths,
        min_size=5,
        max_size=10,
    )
    assert result == [1.0, 1.0, 0.0]
    assert all(isinstance(score, float) for score in result)


def test_size_compliance_materializes_generator_input() -> None:
    lengths = (length for length in [4, 5, 10, 11])

    assert size_compliance(lengths, min_size=5, max_size=10) == [0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("lengths", "min_size", "max_size"),
    [
        ([], 5, 10),
        (np.array([[5, 10]]), 5, 10),
        (["5", "10"], 5, 10),
        ([5, np.nan], 5, 10),
        ([5, np.inf], 5, 10),
        ([5 + 0j], 5, 10),
        ([5], -1, 10),
        ([5], 5, -1),
        ([5], 10, 5),
        ([5], np.nan, 10),
        ([5], 5, np.inf),
        ([5], "5", 10),
    ],
)
def test_size_compliance_returns_empty_list_for_invalid_inputs(
    lengths: object,
    min_size: object,
    max_size: object,
) -> None:
    assert size_compliance(lengths, min_size, max_size) == []


def test_intrachunk_cohesion_returns_score_for_each_chunk() -> None:
    embs = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 0.0]])]

    result = intrachunk_cohesion(embs)

    assert result == pytest.approx([1 / np.sqrt(2), 1.0])
    assert all(isinstance(score, float) for score in result)


@pytest.mark.parametrize(
    "embs",
    [
        [],
        [np.empty((0, 2))],
        [np.empty((2, 0))],
        [np.ones((1, 1, 1))],
        [np.array([[0.0, 0.0]])],
        [np.array([[np.nan, 1.0]])],
        [np.array([[np.inf, 1.0]])],
        [np.array([[1.0 + 1.0j, 0.0]])],
        [np.array([["not", "numeric"]])],
        [np.array([[1.0, 0.0]]), np.empty((0, 2))],
    ],
)
def test_intrachunk_cohesion_returns_empty_list_for_invalid_input(
    embs: list[np.ndarray],
) -> None:
    assert intrachunk_cohesion(embs) == []


def test_contextual_coherence() -> None:
    chunk_embs = np.array([1.0, 0.0])
    context_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = contextual_coherence(chunk_embs, context_embs)
    assert result == pytest.approx(1 / np.sqrt(2))


def test_coreference_integrity_returns_fraction_of_intact_relations() -> None:
    entity_pronoun_spans = [(2, 8), (12, 18), (22, 28)]
    chunk_boundaries = [10, 15]

    assert coreference_integrity(entity_pronoun_spans, chunk_boundaries) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("chunk_boundary", "expected"),
    [
        (10, 1.0),
        (20, 0.0),
    ],
)
def test_coreference_integrity_uses_paper_boundary_interval(
    chunk_boundary: int,
    expected: float,
) -> None:
    assert coreference_integrity([(10, 20)], [chunk_boundary]) == expected


@pytest.mark.parametrize(
    ("entity_pronoun_spans", "chunk_boundaries"),
    [
        ([], [10]),
        ([(2, 8)], []),
    ],
)
def test_coreference_integrity_returns_one_when_nothing_can_be_broken(
    entity_pronoun_spans: list[tuple[int, int]],
    chunk_boundaries: list[int],
) -> None:
    assert coreference_integrity(entity_pronoun_spans, chunk_boundaries) == 1.0


def test_coreference_integrity_materializes_generator_inputs() -> None:
    entity_pronoun_spans = (span for span in [(2, 8), (12, 18)])
    chunk_boundaries = (boundary for boundary in [10, 15])

    assert coreference_integrity(entity_pronoun_spans, chunk_boundaries) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("entity_pronoun_spans", "chunk_boundaries"),
    [
        ([(2.5, 8)], [10]),
        ([(-1, 8)], [10]),
        ([(8, 8)], [10]),
        ([(8, 2)], [10]),
        ([(2, 8, 9)], [10]),
        ([(True, 8)], [10]),
        ([(2, np.nan)], [10]),
        ([(2, 8)], [10.5]),
        ([(2, 8)], [-1]),
        ([(2, 8)], [True]),
        ([(2, 8)], [np.nan]),
        ([(2, 8)], [[10]]),
        (None, [10]),
        ([(2, 8)], None),
    ],
)
def test_coreference_integrity_returns_zero_for_invalid_inputs(
    entity_pronoun_spans: object,
    chunk_boundaries: object,
) -> None:
    assert coreference_integrity(entity_pronoun_spans, chunk_boundaries) == 0.0


def test_boundary_clarity() -> None:
    unc_ppls = np.array([10.0, 8.0, 6.0])
    cond_ppls = np.array([4.0, 1.2])
    result = boundary_clarity(unc_ppls, cond_ppls)
    assert result == pytest.approx([0.5, 0.2])
    assert all(isinstance(score, float) for score in result)


@pytest.mark.parametrize(
    ("uncond_ppls", "cond_ppls"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0, 2.0]), np.array([])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
        (np.array([[1.0, 2.0]]), np.array([1.0])),
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([-1.0])),
        (np.array([1.0, np.inf]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([np.nan])),
    ],
)
def test_boundary_clarity_returns_empty_list_for_invalid_inputs(
    uncond_ppls: np.ndarray,
    cond_ppls: np.ndarray,
) -> None:
    assert boundary_clarity(uncond_ppls, cond_ppls) == []


def test_semantic_dispersion_uses_regularized_centered_gram_matrix() -> None:
    chunk_embs = np.array([[1.0, 3.0], [2.0, 0.0]])

    result = semantic_dispersion(chunk_embs, alpha=0.5)

    # Centering produces [-1, 1] and [1, -1], so det(Gram + 0.5 I) = 2.25.
    assert result == pytest.approx(np.log(2.25) / 2)


def test_semantic_dispersion_rewards_independent_embeddings_over_duplicates() -> None:
    independent_embs = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    duplicate_embs = np.array([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0]])

    assert semantic_dispersion(independent_embs) > semantic_dispersion(duplicate_embs)


def test_semantic_dispersion_increases_with_regularization() -> None:
    chunk_embs = np.array([[1.0, 3.0], [2.0, 0.0]])

    assert semantic_dispersion(chunk_embs, alpha=1.0) > semantic_dispersion(
        chunk_embs,
        alpha=0.1,
    )


def test_semantic_dispersion_supports_one_chunk() -> None:
    assert semantic_dispersion(np.array([[1.0, 3.0]]), alpha=0.5) == pytest.approx(np.log(2.5))


def test_semantic_dispersion_converts_integer_embeddings_before_centering() -> None:
    chunk_embs = np.array([[1, 3], [2, 0]], dtype=np.int64)

    assert semantic_dispersion(chunk_embs, alpha=0.5) == pytest.approx(np.log(2.25) / 2)


def test_semantic_dispersion_returns_zero_when_gram_matrix_overflows() -> None:
    chunk_embs = np.array([[1e308, -1e308], [-1e308, 1e308]])

    assert semantic_dispersion(chunk_embs) == 0.0


@pytest.mark.parametrize(
    ("chunk_embs", "alpha"),
    [
        (np.array([]), 1e-3),
        (np.empty((0, 2)), 1e-3),
        (np.empty((2, 0)), 1e-3),
        (np.ones((1, 1, 1)), 1e-3),
        (np.array([["not", "numeric"]]), 1e-3),
        (np.array([[1.0 + 1.0j, 0.0]]), 1e-3),
        (np.array([[np.nan, 0.0]]), 1e-3),
        (np.array([[np.inf, 0.0]]), 1e-3),
        ([[1.0, 0.0]], 1e-3),
        (np.array([[1.0, 0.0]]), 0.0),
        (np.array([[1.0, 0.0]]), -1.0),
        (np.array([[1.0, 0.0]]), np.nan),
        (np.array([[1.0, 0.0]]), np.inf),
        (np.array([[1.0, 0.0]]), 1.0 + 0.0j),
        (np.array([[1.0, 0.0]]), "0.1"),
    ],
)
def test_semantic_dispersion_returns_zero_for_invalid_inputs(
    chunk_embs: object,
    alpha: object,
) -> None:
    assert semantic_dispersion(chunk_embs, alpha) == 0.0


def test_chunk_score_uses_recommended_logical_independence_weight() -> None:
    assert chunk_score(0.8, 0.2) == pytest.approx(0.38)


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (0.0, -2.0),
        (1.0, 0.8),
    ],
)
def test_chunk_score_supports_boundary_weights(weight: float, expected: float) -> None:
    assert chunk_score(0.8, -2.0, weight) == pytest.approx(expected)


def test_chunk_score_preserves_negative_semantic_dispersion() -> None:
    assert chunk_score(0.5, -1.0, 0.25) == pytest.approx(-0.625)


@pytest.mark.parametrize(
    ("logical_independence", "dispersion", "weight"),
    [
        ("0.8", 0.2, 0.3),
        (0.8, "0.2", 0.3),
        (0.8, 0.2, "0.3"),
        (True, 0.2, 0.3),
        (0.8, False, 0.3),
        (0.8, 0.2, True),
        (np.nan, 0.2, 0.3),
        (0.8, np.inf, 0.3),
        (0.8, 0.2, np.nan),
        (0.8 + 0.0j, 0.2, 0.3),
        (0.8, 0.2, -0.1),
        (0.8, 0.2, 1.1),
        (np.array([0.8]), 0.2, 0.3),
    ],
)
def test_chunk_score_returns_zero_for_invalid_inputs(
    logical_independence: object,
    dispersion: object,
    weight: object,
) -> None:
    assert chunk_score(logical_independence, dispersion, weight) == 0.0


def test_concept_unity_averages_clipped_pairwise_similarities() -> None:
    statements_embs = np.array([[1.0, 0.0], [-1.0, 0.0]])

    assert concept_unity(statements_embs) == pytest.approx(0.5)


def test_concept_unity_returns_one_for_single_statement() -> None:
    statements_embs = np.array([[3.0, 4.0]])

    assert concept_unity(statements_embs) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "statements_embs",
    [
        np.array([[255, 0]], dtype=np.uint8),
        np.array([[60_000, 0]], dtype=np.float16),
        np.array([[1e308, 1e308]]),
    ],
)
def test_concept_unity_avoids_numeric_overflow(statements_embs: np.ndarray) -> None:
    assert concept_unity(statements_embs) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "statements_embs",
    [
        np.array([]),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.ones((1, 1, 1)),
        np.array([[0.0, 0.0]]),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([["not", "numeric"]]),
    ],
)
def test_concept_unity_returns_zero_for_invalid_inputs(
    statements_embs: np.ndarray,
) -> None:
    assert concept_unity(statements_embs) == 0.0


def test_semantic_independence_averages_clipped_paired_similarities() -> None:
    standalone_answer_embs = np.array([[1.0, 0.0], [1.0, 0.0]])
    contextual_answer_embs = np.array([[1.0, 0.0], [-1.0, 0.0]])

    assert metrics.semantic_independence(
        standalone_answer_embs,
        contextual_answer_embs,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("standalone_answer_embs", "contextual_answer_embs"),
    [
        (np.array([]), np.array([])),
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.empty((2, 0)), np.empty((2, 0))),
        (np.ones((1, 1, 1)), np.ones((1, 1, 1))),
        (np.ones((1, 2)), np.ones((2, 2))),
        (np.ones((1, 2)), np.ones((1, 3))),
        (np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])),
        (np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]])),
        (np.array([[np.nan, 0.0]]), np.array([[1.0, 0.0]])),
        (np.array([[1.0, 0.0]]), np.array([[np.inf, 0.0]])),
        (np.array([["not", "numeric"]]), np.array([[1.0, 0.0]])),
        (np.array([[1.0, 0.0]]), np.array([["not", "numeric"]])),
    ],
)
def test_semantic_independence_returns_zero_for_invalid_inputs(
    standalone_answer_embs: np.ndarray,
    contextual_answer_embs: np.ndarray,
) -> None:
    assert metrics.semantic_independence(standalone_answer_embs, contextual_answer_embs) == 0.0


@pytest.mark.parametrize(
    ("standalone_answer_embs", "contextual_answer_embs"),
    [
        (np.array([[255, 255]], dtype=np.uint8), np.array([[255, 255]], dtype=np.uint8)),
        (
            np.array([[60_000, 60_000]], dtype=np.float16),
            np.array([[60_000, 60_000]], dtype=np.float16),
        ),
        (np.array([[1e308, 1e308]]), np.array([[1e308, 1e308]])),
    ],
)
def test_semantic_independence_avoids_numeric_overflow(
    standalone_answer_embs: np.ndarray,
    contextual_answer_embs: np.ndarray,
) -> None:
    assert metrics.semantic_independence(
        standalone_answer_embs,
        contextual_answer_embs,
    ) == pytest.approx(1.0)
