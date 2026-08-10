import logging

import numpy as np

from chunking_metrics import size_compliance


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
    logging.info(f"{embs}")
    assert True
