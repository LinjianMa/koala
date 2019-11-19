"""
This module implements state vector quantum register.
"""

from numbers import Number
from string import ascii_letters as chars
import numpy as np

import tensorbackends

from ..quantum_register import QuantumRegister
from ..gates import tensorize


class StateVectorQuantumRegister(QuantumRegister):
    def __init__(self, nqubit, backend):
        self.backend = tensorbackends.get(backend) if isinstance(backend, str) else backend
        self.state = self.backend.zeros((2,)*nqubit, dtype=complex)
        self.state[(0,)*nqubit] = 1

    @property
    def nqubit(self):
        return self.state.ndim

    def apply_gate(self, gate):
        tensor = tensorize(self.backend, gate.name, *gate.parameters)
        self.state = apply_operator(self.backend, self.state, tensor, gate.qubits)

    def apply_circuit(self, circuit):
        for gate in circuit.gates:
            self.apply_gate(gate)

    def apply_operator(self, operator, qubits):
        self.state = apply_operator(self.backend, self.state, operator, qubits)

    def normalize(self):
        self /= self.norm()

    def norm(self):
        return self.backend.norm(self.state)

    def __imul__(self, a):
        if isinstance(a, Number):
            self.state *= a
            return self
        else:
            return NotImplemented

    def __itruediv__(self, a):
        if isinstance(a, Number):
            self.state /= a
            return self
        else:
            return NotImplemented

    def amplitude(self, bits):
        if len(bits) != self.nqubit:
            raise ValueError('bits number and qubits number do not match')
        return self.state[tuple(bits)]

    def probability(self, bits):
        return np.abs(self.amplitude(bits))**2

    def expectation(self, observable):
        e = 0
        all_terms = ''.join(chars[i] for i in range(self.nqubit))
        einstr = f'{all_terms},{all_terms}->'
        for tensor, qubits in observable:
            state = self.state.copy()
            state = apply_operator(self.backend, state, self.backend.astensor(tensor), qubits)
            e += np.real_if_close(self.backend.einsum(einstr, state, self.state.conj()))
        return e

    def probabilities(self):
        prob_vector = np.real(self.state)**2 + np.imag(self.state)**2
        return [(index, a) for index, a in np.ndenumerate(state) if not np.isclose(a.conj()*a,0)]


def apply_operator(backend, state, operator, axes):
    ndim = state.ndim
    input_state_indices = range(ndim)
    operator_indices = [*axes, *range(ndim, ndim+len(axes))]
    output_state_indices = [*range(ndim)]
    for i, axis in enumerate(axes):
        output_state_indices[axis] = i + ndim
    input_terms = ''.join(chars[i] for i in input_state_indices)
    operator_terms = ''.join(chars[i] for i in operator_indices)
    output_terms = ''.join(chars[i] for i in output_state_indices)
    einstr = f'{input_terms},{operator_terms}->{output_terms}'
    return backend.einsum(einstr, state, operator)
