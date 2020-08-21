import tensorbackends
import numpy as np

from .candecomp import CanonicalDecomp
from .utils import initialize_random_factors


def random(nsite, rank, backend='numpy'):
    backend = tensorbackends.get(backend)
    factors = initialize_random_factors(rank, nsite, backend)
    return CanonicalDecomp(factors, backend)


def basis(nsite, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (1, 2)
    unit_vector = backend.astensor(np.array([1, 0]).reshape(shape))
    factors = [unit_vector for _ in range(nsite)]
    return CanonicalDecomp(factors, backend)


def uniform(nsite, backend='numpy'):
    backend = tensorbackends.get(backend)
    shape = (1, 2)
    unit_vector = backend.astensor(
        np.array([1, 1], dtype=complex).reshape(shape) / np.sqrt(2))
    factors = [unit_vector for _ in range(nsite)]
    return CanonicalDecomp(factors, backend)


def complete_graph_input(nsite, backend='numpy'):
    backend = tensorbackends.get(backend)

    assert nsite % 2 == 0
    nsite_vertex = nsite // 2

    shape = (1, 2)
    unit_vector = backend.astensor(
        np.array([1, 1], dtype=complex).reshape(shape) / np.sqrt(2))
    factors = [[unit_vector] for _ in range(nsite)]

    for num in range(2**nsite_vertex):
        list_binary_str = list(format(num, "b"))
        length = len(list_binary_str)
        for i in range(nsite_vertex - length):
            factors[i].append(
                backend.astensor(
                    np.array([1, 0], dtype=complex).reshape(shape) /
                    np.sqrt(2)))
            factors[i + nsite_vertex].append(
                backend.astensor(np.array([1, 0]).reshape(shape) / np.sqrt(2)))
        for i in range(length):
            j = i + nsite_vertex - length
            if list_binary_str[i] == '0':
                factors[j].append(
                    backend.astensor(
                        np.array([1, 0], dtype=complex).reshape(shape) /
                        np.sqrt(2)))
                factors[j + nsite_vertex].append(
                    backend.astensor(
                        np.array([1, 0]).reshape(shape) / np.sqrt(2)))
            else:
                factors[j].append(
                    backend.astensor(
                        np.array([0, 1], dtype=complex).reshape(shape) /
                        np.sqrt(2)))
                factors[j + nsite_vertex].append(
                    backend.astensor(
                        np.array([0, 1]).reshape(shape) / np.sqrt(2)))

    for i in range(1, len(factors[0])):
        factors[0][i] = -factors[0][i]

    for i in range(nsite):
        factors[i] = backend.vstack(tuple(factors[i]))
        assert factors[i].shape == (1 + 2**nsite_vertex, 2)

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
