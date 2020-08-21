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

    def get_statevector(self):
        out_str = "".join([chr(ord('a') + i) for i in range(self.nsite)])
        einstr = ""
        for i in range(self.nsite - 1):
            einstr += chr(ord('a') + self.nsite) + chr(ord('a') + i) + ','
        einstr += chr(ord('a') + self.nsite) + chr(ord('a') + self.nsite -
                                                   1) + "->" + out_str
        return self.backend.einsum(einstr, *self.factors)

    def apply_circuit_qwalk(self, gates, debug=False):
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
            if gatename.name == 'CX' or gatename.name == 'Uf':

                one_vector = self.backend.astensor(
                    np.array([0, 1], dtype=complex).reshape((1, 2)))
                unit_vector = self.backend.astensor(
                    np.array([1, 1], dtype=complex).reshape(
                        (1, 2)) / np.sqrt(2))
                h_factors = [unit_vector for _ in range(self.nsite)]
                target_factors = [
                    one_vector for _ in range(self.nsite // 2)
                ] + [unit_vector for _ in range(self.nsite // 2)]
                target_factors_trans = [
                    unit_vector for _ in range(self.nsite // 2)
                ] + [one_vector for _ in range(self.nsite // 2)]

                inner_h = inner(self.factors, h_factors, self.backend)
                inner_target = inner(self.factors, target_factors,
                                     self.backend)
                fidel_target_trans = 0.

                nsite_vertices = self.nsite // 2
                N_vertices = 2**nsite_vertices

                fidel_h = (inner_h - inner_target / np.sqrt(N_vertices)) / (
                    1. - 1. / N_vertices)
                fidel_target = (inner_target * np.sqrt(N_vertices) -
                                inner_h) / (np.sqrt(N_vertices) -
                                            1. / np.sqrt(N_vertices))

                print("[similarity to h]", fidel_h)
                print("[similarity to target]", fidel_target)
                print("[similarity to target trans]", fidel_target_trans)

                factors = [[] for _ in range(self.nsite)]
                self.compressed_factors = [[] for _ in range(self.nsite)]
                factors[0].append(fidel_h * h_factors[0])
                factors[0].append(fidel_target * target_factors[0])
                factors[0].append(fidel_target_trans * target_factors_trans[0])
                for i in range(1, self.nsite):
                    factors[i].append(h_factors[i])
                    factors[i].append(target_factors[i])
                    factors[i].append(target_factors_trans[i])

                for i in range(self.nsite):
                    self.compressed_factors[i] = self.backend.vstack(
                        tuple(factors[i]))

                fidel = fidelity(self.factors, self.compressed_factors,
                                 self.backend)
                print(fidel)

                self.fidelity_avg *= fidel
                self.factors = self.compressed_factors

    def apply_circuit(self,
                      gates,
                      rank_threshold=800,
                      hard_compression=False,
                      compress_ratio=0.25,
                      cp_tol=1e-5,
                      cp_maxiter=60,
                      cp_inneriter=20,
                      init_als='random',
                      num_als_init=1,
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
            if self.rank > rank_threshold:
                if hard_compression:
                    target_rank = rank_threshold
                else:
                    target_rank = int(self.rank * compress_ratio)

                self.factors, dtheta = als(self.factors,
                                           self.backend,
                                           target_rank,
                                           tol=cp_tol,
                                           max_iter=cp_maxiter,
                                           inner_iter=cp_inneriter,
                                           init_als=init_als,
                                           num_als_init=num_als_init,
                                           debug=debug)
                self.theta += dtheta
                self.fidelity_lower = np.cos(self.theta)
                self.fidelity_avg *= np.cos(dtheta)

    def apply_rankone_gate_inplace(self, gate):
        for (i, qubit) in enumerate(gate.qubits):
            self.factors[qubit] = self.factors[qubit] @ gate.operators[i]

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

    def apply_multirank_gate(self, gate):
        factors = [[] for _ in range(self.nsite)]
        for rank_one_gate in gate.rank_one_gates:
            rank_one_factors = self.apply_rankone_gate(rank_one_gate)
            for i in range(self.nsite):
                factors[i].append(rank_one_factors[i])

        for i in range(self.nsite):
            factors[i] = self.backend.vstack(tuple(factors[i]))

        return factors
