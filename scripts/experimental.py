import logging

import numpy as np

from chunking_metrics import intrachunk_cohesion

logging.basicConfig(level=logging.DEBUG)

embs = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 0.0]])]

result = intrachunk_cohesion(embs)

print(np.cos())

print(result, "fgdfd")
