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
        self.prev_factors = None
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

    def apply_circuit(self,
                      gates,
                      rank_threshold=800,
                      hard_compression=True,
                      compress_ratio=0.25,
                      cp_tol=1e-5,
                      cp_maxiter=60,
                      cp_inneriter=20,
                      init_als='random',
                      num_als_init=1,
                      use_prev_factor=False,
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

                if use_prev_factor:
                    set_prev_factor = self.prev_factors
                else:
                    set_prev_factor = None
                self.factors, dtheta = als(self.factors,
                                           self.backend,
                                           target_rank,
                                           tol=cp_tol,
                                           max_iter=cp_maxiter,
                                           inner_iter=cp_inneriter,
                                           init_als=init_als,
                                           num_als_init=num_als_init,
                                           prev_factors=set_prev_factor,
                                           debug=debug)

                self.prev_factors = self.factors

                self.theta += dtheta
                self.fidelity_lower = np.cos(self.theta)
                self.fidelity_avg *= np.cos(dtheta)

    def apply_circuit_direct_cpd(self, gates, rank_threshold=800, debug=False):
        layer_num = 0
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
            if self.rank > rank_threshold:  # and gatename.name == "Uf":
                if debug:
                    print(f"Layer number is: {layer_num}")
                layer_num += 1

                factors_list = extract_rank1_tensors(self.factors)
                K = len(factors_list)
                i = 0
                while i < K:
                    indices_to_remove = []
                    for j in range(i + 1, K):
                        alpha = inner(factors_list[i], factors_list[j],
                                      self.backend)
                        beta = norm(factors_list[i], self.backend)
                        gamma = norm(factors_list[j], self.backend)
                        if beta == 0:
                            break
                        if gamma == 0:
                            indices_to_remove.append(j)
                            continue
                        if 1. - 1e-5 < abs(alpha / (gamma * beta)) < 1. + 1e-5:
                            factors_list[i][0] = (
                                1 + alpha / beta**2) * factors_list[i][0]
                            indices_to_remove.append(j)

                    for index in reversed(indices_to_remove):
                        del factors_list[index]
                    i += 1
                    K = len(factors_list)

                # remove the elements where norm is 0, get all the rank-1 tensor norm
                for index in reversed(range(len(factors_list))):
                    if norm(factors_list[index], self.backend) == 0:
                        del factors_list[index]
                nrm_list = []
                for index in range(len(factors_list)):
                    nrm_list.append(
                        (index, norm(factors_list[index], self.backend)))
                nrm_list = sorted(nrm_list, key=lambda tup: tup[1])
                nrm_list.reverse()

                outfactors = [[] for _ in range(self.nsite)]
                for i in range(self.nsite):
                    for j in range(min(rank_threshold, len(nrm_list))):
                        outfactors[i].append(factors_list[nrm_list[j][0]][i])
                    outfactors[i] = self.backend.vstack(tuple(outfactors[i]))

                self.fidelity_lower = 0
                fidel = fidelity(self.factors, outfactors, self.backend)
                if debug:
                    print(f"Fidelity is: {fidel}")
                self.fidelity_avg *= fidel
                self.factors = outfactors

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
