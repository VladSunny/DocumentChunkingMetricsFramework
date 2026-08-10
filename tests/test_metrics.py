import numpy as np
import pytest

from chunking_metrics import (
    boundary_clarity,
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


def test_intrachunk_cohesion() -> None:
    embs = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 0.0]])]
    result = intrachunk_cohesion(embs)
    assert result == pytest.approx((1 / np.sqrt(2) + 1) / 2)


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
