"""
This module defines the gates used for the CP format simulator.
"""

from functools import lru_cache

from .. import tensors


class MultiRankGate(object):
    def __init__(self, qubits_list, operators_list):
        assert len(qubits_list) == len(operators_list)
        self.rank = len(qubits_list)
        self.qubits_list = qubits_list
        self.operators_list = operators_list

        self.rank_one_gates = [
            RankOneGate(qubits_list[i], operators_list[i])
            for i in range(self.rank)
        ]


class RankOneGate(object):
    def __init__(self, qubits, operators):
        self.qubits = qubits
        self.operators = operators


class SwapGate(object):
    def __init__(self, qubits):
        assert len(qubits) == 2
        self.qubits = qubits


def get_gate(backend, gate):
    if gate.name not in _GATES:
        raise ValueError(f"{gate.name} gate is not supported")
    return _GATES[gate.name](backend, tuple(gate.qubits), *gate.parameters)


_GATES = {}


def _register(func):
    _GATES[func.__name__] = func
    return func


@_register
@lru_cache(maxsize=None)
def SWAP(backend, qubits):
    return SwapGate(qubits)


@_register
@lru_cache(maxsize=None)
def H(backend, qubits):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.H())
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def Hs(backend, qubits):
    """
    Note: this is not a elementary gate. Used for collapsing the CP ranks
    """
    operator = backend.astensor(tensors.H())
    operators = [operator for _ in qubits]
    return RankOneGate(qubits, operators)


@_register
@lru_cache(maxsize=None)
def S(backend, qubits):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.S())
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def E1(backend, qubits):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.E1())
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def E2(backend, qubits):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.E2())
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def R(backend, qubits, theta):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.R(theta))
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def FLIP(backend, qubits):
    assert len(qubits) == 1
    operator = -backend.identity(2)
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def CR(backend, qubits, theta):
    assert len(qubits) == 2
    qubits_list = [[qubits[0]], qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    R = backend.astensor(tensors.R(theta))
    operators_list = [[E1], [E2, R]]
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def CRs(backend, qubits, *theta_list):
    """
    Note: this is not a elementary gate. Used for collapsing the CP ranks
    """
    assert len(qubits) >= 2
    qubits_list = [[qubits[0]], qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    Rs = [backend.astensor(tensors.R(theta)) for theta in theta_list]
    operators_list = [[E1], [E2] + Rs]
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def Uf(backend, qubits, *marked_states):
    """
    Uf gate used in the Grover's algorithm.
        Ref: https://qiskit.org/textbook/ch-algorithms/grover.html#3qubits
    """
    assert len(qubits) >= 2
    qubits_list = [[]] + [qubits for _ in marked_states]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    operators_list = [[]]
    for state in marked_states:
        assert len(state) == len(qubits)
        operators = []
        for bit in state:
            if bit == 0:
                operator = E1
            elif bit == 1:
                operator = E2
            operators.append(operator)
        operators[0] = -2 * operators[0]
        operators_list.append(operators)
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def U0(backend, qubits):
    assert len(qubits) >= 2
    marked_states = [tuple(0 for _ in range(len(qubits)))]
    return Uf(backend, qubits, *marked_states)
