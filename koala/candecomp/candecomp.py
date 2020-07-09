"""
This module implements canonical format quantum register.
"""

import numpy as np
import copy
import tensorbackends

from .utils import *
from .als import als
from ..quantum_state import QuantumState
from .gates import MultiRankGate, RankOneGate, SwapGate, get_gate


class CanonicalDecomp(QuantumState):
    def __init__(self, factors, backend):
        self.backend = tensorbackends.get(backend)
        self.factors = factors
        self.theta = 0.
        self.fidelity_lower = 1.
        self.fidelity_avg = 1.

    @property
    def nsite(self):
        return len(self.factors)

    @property
    def rank(self):
        return self.factors[0].shape[0]

    def copy(self):
        factors = copy.deepcopy(self.factors)
        return CanonicalDecomp(factors, self.backend)

    def norm(self):
        return utils.norm(self.factors, self.backend)

    def inner(self, other):
        return utils.inner(self.factors, self.factors, self.backend)

    def flatten_factors(self):
        if len(self.factors[0].shape) == 2:
            return self.factors
        elif len(self.factors[0].shape) == 3:
            return [factor.reshape(self.rank, 4) for factor in self.factors]

    def get_statevector(self):
        factors = self.flatten_factors()
        out_str = "".join([chr(ord('a') + i) for i in range(self.nsite)])
        einstr = ""
        for i in range(self.nsite - 1):
            einstr += chr(ord('a') + self.nsite) + chr(ord('a') + i) + ','
        einstr += chr(ord('a') + self.nsite) + chr(ord('a') + self.nsite -
                                                   1) + "->" + out_str
        return self.backend.einsum(einstr, *factors)

    def apply_circuit(self,
                      gates,
                      rank_threshold=800,
                      compress_ratio=0.25,
                      cp_tol=1e-5,
                      cp_maxiter=60,
                      cp_inneriter=20,
                      init_als='random',
                      mode='state',
                      debug=False):
        if mode == 'state':
            self.apply_circuit_state(gates,
                                     rank_threshold=rank_threshold,
                                     compress_ratio=compress_ratio,
                                     cp_tol=cp_tol,
                                     cp_maxiter=cp_maxiter,
                                     cp_inneriter=cp_inneriter,
                                     init_als=init_als,
                                     debug=debug)
        elif mode == 'operator':
            self.apply_circuit_operator(gates,
                                        rank_threshold=rank_threshold,
                                        compress_ratio=compress_ratio,
                                        cp_tol=cp_tol,
                                        cp_maxiter=cp_maxiter,
                                        cp_inneriter=cp_inneriter,
                                        init_als=init_als,
                                        debug=debug)

    def apply_circuit_state(self,
                            gates,
                            rank_threshold=800,
                            compress_ratio=0.25,
                            cp_tol=1e-5,
                            cp_maxiter=60,
                            cp_inneriter=20,
                            init_als='random',
                            debug=False):
        for gatename in gates:
            gate = get_gate(self.backend, gatename)
            if isinstance(gate, RankOneGate):
                self.apply_rankone_gate_inplace(gate)
            elif isinstance(gate, SwapGate):
                self.apply_swap_gate(gate.qubits)
            elif isinstance(gate, MultiRankGate):
                self.factors = self.apply_multirank_gate(gate)
            if debug:
                print(
                    f"After applying gate: {gatename}, CP rank is {self.rank}")
            # apply CP compression
            if self.rank >= rank_threshold:
                self.factors, dtheta = als(self.factors,
                                           self.backend,
                                           int(self.rank * compress_ratio),
                                           tol=cp_tol,
                                           max_iter=cp_maxiter,
                                           inner_iter=cp_inneriter,
                                           init_als=init_als,
                                           debug=debug)
                self.theta += dtheta
                self.fidelity_lower = np.cos(self.theta)
                self.fidelity_avg *= np.cos(dtheta)

    def apply_circuit_operator(self,
                               gates,
                               rank_threshold=800,
                               compress_ratio=0.25,
                               cp_tol=1e-5,
                               cp_maxiter=60,
                               cp_inneriter=20,
                               init_als='random',
                               debug=False):
        for gatename in gates:
            gate = get_gate(self.backend, gatename)
            if isinstance(gate, RankOneGate):
                self.apply_rankone_gate_operator_inplace(gate)
            elif isinstance(gate, SwapGate):
                self.apply_swap_gate(gate.qubits)
            elif isinstance(gate, MultiRankGate):
                self.factors = self.apply_multirank_gate_operator(gate)
            if debug:
                print(
                    f"After applying gate: {gatename}, CP rank is {self.rank}")
            # apply CP compression
            if self.rank >= rank_threshold:
                factors = [
                    factor.reshape(self.rank, 4) for factor in self.factors
                ]
                factors, dtheta = als(factors,
                                      self.backend,
                                      int(rank_threshold),
                                      tol=cp_tol,
                                      max_iter=cp_maxiter,
                                      inner_iter=cp_inneriter,
                                      init_als=init_als,
                                      debug=debug)
                self.factors = [
                    factor.reshape(factor.shape[0], 2, 2) for factor in factors
                ]
                self.theta += dtheta
                self.fidelity_lower = np.cos(self.theta)
                self.fidelity_avg *= np.cos(dtheta)

    def apply_rankone_gate_inplace(self, gate):
        for (i, qubit) in enumerate(gate.qubits):
            self.factors[qubit] = self.factors[qubit] @ gate.operators[i]

    def apply_rankone_gate_operator_inplace(self, gate):
        for (i, qubit) in enumerate(gate.qubits):
            self.factors[qubit] = self.backend.einsum("ijk,kl->ijl",
                                                      self.factors[qubit],
                                                      gate.operators[i])

    def apply_swap_gate(self, qubits):
        assert len(qubits) == 2
        temp_factor = self.factors[qubits[0]]
        self.factors[qubits[0]] = self.factors[qubits[1]]
        self.factors[qubits[1]] = temp_factor

    def apply_rankone_gate(self, gate):
        factors = copy.deepcopy(self.factors)
        for (i, qubit) in enumerate(gate.qubits):
            factors[qubit] = factors[qubit] @ gate.operators[i]
        return factors

    def apply_rankone_gate_operator(self, gate):
        factors = copy.deepcopy(self.factors)
        for (i, qubit) in enumerate(gate.qubits):
            factors[qubit] = self.backend.einsum("ijk,kl->ijl", factors[qubit],
                                                 gate.operators[i])
        return factors

    def apply_multirank_gate(self, gate):
        factors = [[] for _ in range(self.nsite)]
        for rank_one_gate in gate.rank_one_gates:
            rank_one_factors = self.apply_rankone_gate(rank_one_gate)
            for i in range(self.nsite):
                factors[i].append(rank_one_factors[i])

        for i in range(self.nsite):
            factors[i] = self.backend.vstack(tuple(factors[i]))
        return factors

    def apply_multirank_gate_operator(self, gate):
        rank = self.rank
        factors = [[] for _ in range(self.nsite)]
        for rank_one_gate in gate.rank_one_gates:
            rank_one_factors = self.apply_rankone_gate(rank_one_gate)
            for i in range(self.nsite):
                factors[i].append(rank_one_factors[i].reshape(rank, 4))

        for i in range(self.nsite):
            factors[i] = self.backend.vstack(tuple(factors[i]))
            factors[i] = factors[i].reshape(factors[i].shape[0], 2, 2)
        return factors
