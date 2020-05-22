import numpy as np
import tensorbackends

from .candecomp import CanonicalDecomp


def random(nsite, rank, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (rank, 2)
    factors = [
        backend.random.uniform(-1, 1, shape) +
        1j * backend.random.uniform(-1, 1, shape) for _ in range(nsite)
    ]
    return CanonicalDecomp(factors, backend)
