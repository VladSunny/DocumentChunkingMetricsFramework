import numpy as np
import pytest

from chunking_metrics import contextual_coherence, intrachunk_cohesion, size_compliance


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
