# import ctf
import numpy as np
import tensorbackends as tb


def solve_sys(g, rhs, backend):
    if isinstance(g.unwrap(), np.ndarray):
        return np.linalg.solve(g, rhs)
    elif isinstance(g.unwrap(), ctf.core.tensor):
        rhs_t = rhs.transpose()
        out_t = backend.solve_spd(g.transpose(), rhs_t)
        out = out_t.transpose()
        return out


def initialize_random_factors(rank, nsite, backend):
    shape = (rank, 2)
    return [
        backend.random.uniform(-1, 1, shape) +
        1j * backend.random.uniform(-1, 1, shape) for _ in range(nsite)
    ]


def initialize_hosvd(factors, index, backend):
    hadamard_prod = backend.ones(
        (factors[0].shape[0], factors[0].shape[0])).astype('complex128')
    for i in range(0, index):
        hadamard_prod *= factors[i] @ backend.transpose(factors[i].conj())
    for i in range(index + 1, len(factors)):
        hadamard_prod *= factors[i] @ backend.transpose(factors[i].conj())
    normal_mat = backend.einsum("ka,lb,kl->ab", factors[i], factors[i],
                                hadamard_prod)
    U, s, _ = backend.svd(normal_mat)
    return backend.transpose(U)


def norm(factors, backend):
    return np.sqrt(inner(factors, factors, backend))


def inner(factors, other_factors, backend):
    hadamard_prod = factors[0] @ backend.transpose(other_factors[0].conj())
    for i in range(1, len(factors)):
        hadamard_prod *= factors[i] @ backend.transpose(
            other_factors[i].conj())
    return backend.sum(hadamard_prod)


def fitness(factors, compressed_factors, backend):
    normsq_factors = inner(factors, factors, backend)
    normsq_compressed_factors = inner(compressed_factors, compressed_factors,
                                      backend)
    inner_product = inner(factors, compressed_factors, backend)
    residual = np.sqrt(normsq_factors + normsq_compressed_factors -
                       2 * inner_product)
    norm_factors = np.sqrt(normsq_factors)
    fit = 1. - residual / norm_factors
    return fit


def fidelity(factors, compressed_factors, backend):
    return inner(factors, compressed_factors, backend) / (
        norm(factors, backend) * norm(compressed_factors, backend))
