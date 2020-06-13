import tensorbackends
import numpy as np

from .candecomp import CanonicalDecomp
from .utils import initialize_random_factors


def random(nsite, rank, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (rank, 2)
    factors = initialize_random_factors(shape, nsite, backend)
    return CanonicalDecomp(factors, backend)


def basis(nsite, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (1, 2)
    unit_vector = backend.astensor(np.array([1, 0]).reshape(shape))
    factors = [unit_vector for _ in range(nsite)]
    return CanonicalDecomp(factors, backend)


def identity(nsite, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (1, 2, 2)
    I = backend.astensor(np.array([[1, 0], [0, 1]]).reshape(shape))
    factors = [I for _ in range(nsite)]
    return CanonicalDecomp(factors, backend)


def rectangular_pulse(nsite, backend='numpy'):
    assert nsite % 2 == 0
    shape = (1, 2)

    backend = tensorbackends.get(backend)
    factors = []
    for _ in range(int(nsite / 2)):
        factors.append(backend.astensor(np.array([1, 0]).reshape(shape)))
    for _ in range(int(nsite / 2), nsite):
        factors.append(backend.astensor(np.array([1, 1]).reshape(shape)))

    return CanonicalDecomp(factors, backend)
