import numpy as np
import pytest

import chunking_metrics.metrics as metrics
from chunking_metrics.metrics import (
    boundary_clarity,
    concept_unity,
    contextual_coherence,
    intrachunk_cohesion,
    size_compliance,
)


def test_size_compliance() -> None:
    lengths = np.array([5, 10, 20])
    result = size_compliance(
        lengths=lengths,
        min_size=5,
        max_size=10,
    )
    assert result == 2 / 3


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


def test_boundary_clarity() -> None:
    unc_ppls = np.array([10.0, 8.0, 6.0])
    cond_ppls = np.array([4.0, 3.0])
    result = boundary_clarity(unc_ppls, cond_ppls)
    assert result == 0.5


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
def test_boundary_clarity_returns_zero_for_invalid_inputs(
    uncond_ppls: np.ndarray,
    cond_ppls: np.ndarray,
) -> None:
    assert boundary_clarity(uncond_ppls, cond_ppls) == 0.0


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
