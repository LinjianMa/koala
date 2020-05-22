import numpy as np


def initialize_random_factors(rank, nsite, backend):
    shape = (rank, 2)
    return [
        backend.random.uniform(-1, 1, shape) +
        1j * backend.random.uniform(-1, 1, shape) for _ in range(nsite)
    ]


def norm(factors, backend):
    return np.sqrt(inner(factors, factors, backend))


def inner(factors, other_factors, backend):
    hadamard_prod = factors[0] @ backend.transpose(other_factors[0].conj())
    for i in range(1, len(factors)):
        hadamard_prod *= factors[i] @ backend.transpose(
            other_factors[i].conj())
    return backend.sum(hadamard_prod)
