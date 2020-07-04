"""
This module implements ALS algorithms for the CP compression.
"""
import numpy as np
import tensorbackends
import numpy.linalg as la
from .utils import initialize_random_factors, fitness, fidelity, solve_sys


class ALSOptimizer(object):
    def __init__(self,
                 factors,
                 backend,
                 rank,
                 init_als='random',
                 compressed_factors=None):
        self.backend = backend
        self.factors = factors
        self.rank = rank
        self.nsite = len(factors)
        assert self.nsite >= 3
        self.compressed_factors = compressed_factors

        if init_als == 'factors' and compressed_factors is None:
            self.compressed_factors = [
                self.factors[i][:rank, :] for i in range(self.nsite)
            ]
        elif init_als == 'random' and compressed_factors is None:
            self.compressed_factors = initialize_random_factors(
                rank, self.nsite, backend)

    def step(self):
        for i in range(self.nsite):
            m = mttkrp(self.factors, self.compressed_factors, i)
            g = gram(self.compressed_factors,
                     i) + 1e-7 * self.backend.identity(self.rank)
            x = self.backend.astensor(solve_sys(g, m, self.backend))
            self.compressed_factors[i] = x


def als_onestep(factors, compressed_factors, backend, rank):
    optimizer = ALSOptimizer(factors,
                             backend,
                             rank,
                             compressed_factors=compressed_factors)
    optimizer.step()
    return optimizer.compressed_factors


def als(factors,
        backend,
        rank,
        tol=1e-7,
        max_iter=200,
        inner_iter=10,
        init_als='random',
        debug=False):
    optimizer = ALSOptimizer(factors, backend, rank, init_als)
    num_iter = 0
    fidel_old, fidel = 0., 0.

    while num_iter < max_iter:

        for _ in range(inner_iter):
            optimizer.step()
            num_iter += 1

        # calculate the fidelity
        fidel = fidelity(optimizer.factors, optimizer.compressed_factors,
                         backend)
        if debug:
            print(
                f"ALS iterations: at interations {num_iter} the fidelity is {fidel}."
            )

        if abs(fidel - fidel_old) < tol:
            return optimizer.compressed_factors, np.arccos(fidel)
        fidel_old = fidel

    return optimizer.compressed_factors, np.arccos(fidel)


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
