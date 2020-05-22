"""
This module implements ALS algorithms for the CP compression.
"""
import numpy as np
import tensorbackends
import numpy.linalg as la
from .utils import initialize_random_factors, norm, inner


class ALSOptimizer(object):
    def __init__(self, factors, backend, rank):
        self.backend = backend
        self.factors = factors
        self.rank = rank
        self.nsite = len(factors)
        assert self.nsite >= 3
        self.compressed_factors = initialize_random_factors(
            rank, self.nsite, backend)

    def step(self):
        for i in range(self.nsite):
            m = mttkrp(self.factors, self.compressed_factors, i)
            g = gram(self.compressed_factors, i)
            x = self.backend.astensor(la.solve(g, m))
            self.compressed_factors[i] = x


def als(factors,
        backend,
        rank,
        tol=1e-7,
        max_iter=200,
        inner_iter=10,
        debug=False):
    optimizer = ALSOptimizer(factors, backend, rank)
    num_iter = 0
    fitness_old, fitness = 0., 0.

    while num_iter < max_iter:

        for _ in range(inner_iter):
            optimizer.step()
            num_iter += 1

        # calculate the fitness
        normsq_factors = inner(optimizer.factors, optimizer.factors, backend)
        normsq_compressed_factors = inner(optimizer.compressed_factors,
                                          optimizer.compressed_factors,
                                          backend)
        inner_product = inner(optimizer.factors, optimizer.compressed_factors,
                              backend)
        residual = np.sqrt(normsq_factors + normsq_compressed_factors -
                           2 * inner_product)
        norm_factors = np.sqrt(normsq_factors)
        fitness = 1. - residual / norm_factors

        if debug:
            print(
                f"ALS iterations: at interations {num_iter} the fitness is {fitness}."
            )

        if abs(fitness - fitness_old) < tol:
            return optimizer.compressed_factors
        fitness_old = fitness

    return optimizer.compressed_factors


def mttkrp(factors, compressed_factors, i):
    nsite = len(factors)
    if i == 0:
        hadamard_prod = factors[1] @ compressed_factors[1].conj().transpose()
        for j in range(2, nsite):
            hadamard_prod *= factors[j] @ compressed_factors[j].conj(
            ).transpose()
        return hadamard_prod.transpose() @ factors[0]
    else:
        hadamard_prod = factors[0] @ compressed_factors[0].conj().transpose()
        for j in range(1, i):
            hadamard_prod *= factors[j] @ compressed_factors[j].conj(
            ).transpose()
        for j in range(i + 1, nsite):
            hadamard_prod *= factors[j] @ compressed_factors[j].conj(
            ).transpose()
        return hadamard_prod.transpose() @ factors[i]


def gram(factors, i):
    nsite = len(factors)
    if i == 0:
        hadamard_prod = factors[1] @ factors[1].conj().transpose()
        for j in range(2, nsite):
            hadamard_prod *= factors[j] @ factors[j].conj().transpose()
    else:
        hadamard_prod = factors[0] @ factors[0].conj().transpose()
        for j in range(1, i):
            hadamard_prod *= factors[j] @ factors[j].conj().transpose()
        for j in range(i + 1, nsite):
            hadamard_prod *= factors[j] @ factors[j].conj().transpose()
    return hadamard_prod.transpose()
