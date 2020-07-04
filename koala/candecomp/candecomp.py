"""
This module implements canonical format quantum register.
"""

import numpy as np
import copy, math, random
import tensorbackends

from .utils import *
from .als import als, als_onestep
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

    def apply_circuit_implicit(self,
                               gates,
                               rank_threshold=1024,
                               cp_maxiter=60,
                               compress_ratio=0.5):
        num_sampled_multirank_gates = int(math.log(rank_threshold, 2))
        assert num_sampled_multirank_gates == math.log(rank_threshold, 2)

        # extract multirankgate
        multirank_indices = []
        for i, gatename in enumerate(gates):
            gate = get_gate(self.backend, gatename)
            if isinstance(gate, MultiRankGate):
                multirank_indices.append(i)
        assert len(multirank_indices) >= num_sampled_multirank_gates

        out_factors = initialize_random_factors(
            int(rank_threshold * compress_ratio), self.nsite, self.backend)
        for iter in range(cp_maxiter):

            nrm = norm(out_factors, self.backend)
            nrm_input = norm(self.factors, self.backend)

            # out_factors[-1] = nrm_input / nrm * out_factors[-1]
            # nrm = norm(out_factors, self.backend)
            print(nrm, nrm_input)

            #get sampled gates
            sampled_gates = []
            sampled_multirank_indices = random.sample(
                multirank_indices, k=num_sampled_multirank_gates)
            for i, gatename in enumerate(gates):
                gate = get_gate(self.backend, gatename)
                if isinstance(
                        gate,
                        MultiRankGate) and i not in sampled_multirank_indices:
                    # sampled_rank_one_index = [1]#random.choices(list(range(gate.rank)))
                    # sampled_gates.append(gate.rank_one_gates[sampled_rank_one_index[0]])
                    sampled_gates.append(gate.rank_one_gates[0] +
                                         gate.rank_one_gates[1])
                else:
                    sampled_gates.append(gate)

            random_factors = apply_gates(sampled_gates, self.nsite,
                                         self.backend, self.factors)

            nrm = norm(random_factors, self.backend)
            nrm_self_factors = norm(self.factors, self.backend)
            print("norm of the random factors", nrm, nrm_self_factors)

            for _ in range(1):
                out_factors = als_onestep(random_factors, out_factors,
                                          self.backend,
                                          int(rank_threshold * compress_ratio))
            # for factor in out_factors:
            #     for i in range(factor.shape[1]):
            #         if np.linalg.norm(factor[:,i]) < 1e-7:
            #             shape = [int(rank_threshold * compress_ratio)]
            #             factor[:,i] = self.backend.random.uniform(-1, 1, shape) + 1j * self.backend.random.uniform(-1, 1, shape)
            print(out_factors)

        self.factors = out_factors

    def apply_circuit(self,
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
                self.factors = apply_swap_gate(gate.qubits, self.factors)
            elif isinstance(gate, MultiRankGate):
                self.factors = apply_multirank_gate(gate, self.nsite,
                                                    self.backend, self.factors)
            if debug:
                print(
                    f"After applying gate: {gatename}, CP rank is {self.rank}")
            # apply CP compression
            if self.rank > rank_threshold:
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

    def apply_rankone_gate_inplace(self, gate):
        for (i, qubit) in enumerate(gate.qubits):
            self.factors[qubit] = self.factors[qubit] @ gate.operators[i]


def apply_swap_gate(qubits, in_factors):
    factors = copy.deepcopy(in_factors)
    assert len(qubits) == 2
    temp_factor = factors[qubits[0]]
    factors[qubits[0]] = factors[qubits[1]]
    factors[qubits[1]] = temp_factor
    return factors


def apply_rankone_gate(gate, in_factors):
    factors = copy.deepcopy(in_factors)
    for (i, qubit) in enumerate(gate.qubits):
        nrm = np.linalg.norm(factors[qubit])
        factors[qubit] = factors[qubit] @ gate.operators[i]
        nrm2 = np.linalg.norm(factors[qubit])
        print(gate.operators[i])
        print("diffnorm", nrm - nrm2)
    return factors


def apply_multirank_gate(gate, nsite, backend, in_factors):
    factors = [[] for _ in range(nsite)]
    for rank_one_gate in gate.rank_one_gates:
        rank_one_factors = apply_rankone_gate(rank_one_gate, in_factors)
        for i in range(nsite):
            factors[i].append(rank_one_factors[i])

    for i in range(nsite):
        factors[i] = backend.vstack(tuple(factors[i]))
    return factors


def apply_gates(gates, nsite, backend, factors):
    for gate in gates:
        if isinstance(gate, RankOneGate):
            factors = apply_rankone_gate(gate, factors)
        elif isinstance(gate, SwapGate):
            factors = apply_swap_gate(gate.qubits, factors)
        elif isinstance(gate, MultiRankGate):
            factors = apply_multirank_gate(gate, len(factors), backend,
                                           factors)
        print(f"After applying gate: {gate}, CP rank is {factors[0].shape[0]}")
    return factors
