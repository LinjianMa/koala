import tensorbackends
import numpy as np

from .candecomp import CanonicalDecomp
from .utils import initialize_random_factors


def random(nsite, rank, backend='numpy'):
    backend = tensorbackends.get(backend)
    factors = initialize_random_factors(rank, nsite, backend)
    return CanonicalDecomp(factors, backend)


def rectangular_pulse(nsite, backend='numpy'):
    assert nsite % 2 == 0
    shape = (1, 2)

    backend = tensorbackends.get(backend)
    factors = []
    for _ in range(int(nsite / 2)):
        factors.append(backend.astensor(np.array([0, 1]).reshape(shape)))
    for _ in range(int(nsite / 2), nsite):
        factors.append(backend.astensor(np.array([1, 1]).reshape(shape)))

    return CanonicalDecomp(factors, backend)
