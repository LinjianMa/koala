"""
This module implements canonical format quantum register.
"""

from numbers import Number
import numpy as np
import copy

import tensorbackends

from ..quantum_state import QuantumState
from .gates import MultiRankGate, RankOneGate, SwapGate, get_gate


class CanonicalDecomp(QuantumState):
    def __init__(self, factors, backend):
        self.backend = tensorbackends.get(backend)
        self.factors = factors

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
        return np.sqrt(self.inner(self))

    def inner(self, other):
        hadamard_prod = self.factors[0] @ self.backend.transpose(
            other.factors[0].conj())
        for i in range(1, self.nsite):
            hadamard_prod *= self.factors[i] @ self.backend.transpose(
                other.factors[i].conj())
        return self.backend.sum(hadamard_prod)

    def get_statevector(self):
        out_str = "".join([chr(ord('a') + i) for i in range(self.nsite)])
        einstr = ""
        for i in range(self.nsite - 1):
            einstr += chr(ord('a') + self.nsite) + chr(ord('a') + i) + ','
        einstr += chr(ord('a') + self.nsite) + chr(ord('a') + self.nsite -
                                                   1) + "->" + out_str
        return self.backend.einsum(einstr, *self.factors)

    def apply_circuit(self, gates, debug=False):
        for gate in gates:
            if debug:
                print(f"Applying gate: {gate}, CP rank is {self.rank}")
            gate = get_gate(self.backend, gate)
            if isinstance(gate, RankOneGate):
                self.apply_rankone_gate_inplace(gate)
            elif isinstance(gate, SwapGate):
                self.apply_swap_gate(gate.qubits)
            elif isinstance(gate, MultiRankGate):
                self.factors = self.apply_multirank_gate(gate)

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
            factors[i] = self.backend.concatenate(factors[i], axis=0)

        return factors
