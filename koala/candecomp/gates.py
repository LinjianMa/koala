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
def Ry(backend, qubits, theta):
    assert len(qubits) == 1
    operator = backend.astensor(tensors.Ry(theta))
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def FLIP(backend, qubits):
    assert len(qubits) == 1
    operator = -backend.identity(2)
    return RankOneGate(qubits, [operator])


@_register
@lru_cache(maxsize=None)
def CRi(backend, qubits, theta):
    assert len(qubits) == 2
    qubits_list = [[qubits[0]], qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    R = backend.astensor(tensors.R(theta))
    operators_list = [[E1], [E2, R]]
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def CX(backend, qubits):
    assert len(qubits) == 2
    qubits_list = [[qubits[0]], qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    X = backend.astensor(tensors.X())
    operators_list = [[E1], [E2, X]]
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
def CRs_inv(backend, qubits, *theta_list):
    """
    Note: this is not a elementary gate. Used for collapsing the CP ranks
    """
    assert len(qubits) >= 2
    qubits_list = [[qubits[0]], qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    Rs = [backend.astensor(tensors.Rinv(theta)) for theta in theta_list]
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
def General_control(backend, qubits, *gatelist):
    """
    General controlled gate. gatelist can be (0, 1, 'X').
    """
    assert len(qubits) >= 2
    qubits_list = [[], qubits, qubits]
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    X = backend.astensor(tensors.X())
    Z = backend.astensor(tensors.Z())
    H = backend.astensor(tensors.H())
    I = backend.astensor(tensors.I())
    assert len(gatelist) == len(qubits)

    operators_list = [[]]
    operators1 = []
    operators2 = []
    for gate in gatelist:
        if gate == 0:
            operators1.append(E1)
            operators2.append(E1)
        elif gate == 1:
            operators1.append(E2)
            operators2.append(E2)
        elif gate == "X":
            operators1.append(X)
            operators2.append(I)
        elif gate == "Z":
            operators1.append(Z)
            operators2.append(I)
        elif gate == "H":
            operators1.append(H)
            operators2.append(I)
        operators2[-1] = -operators2[-1]
    operators_list.append(operators1)
    operators_list.append(operators2)
    return MultiRankGate(qubits_list, operators_list)


def operators_rotation(backend, qubits, mode='L'):
    assert len(qubits) >= 2
    if mode == 'L':
        E = backend.astensor(tensors.E1())
    elif mode == 'R':
        E = backend.astensor(tensors.E2())
    XI = backend.astensor(tensors.X() - tensors.I())
    X = backend.astensor(tensors.X())
    I = backend.astensor(tensors.I())

    n = len(qubits)
    qubits_list = [list(qubits) for _ in range(n)]
    operators_list = []
    for i in range(n - 1):
        operators = []
        for j in range(i):
            operators.append(I)
        operators.append(XI)
        for j in range(i + 1, n):
            operators.append(E @ X)
        operators_list.append(operators)

    operators = []
    for j in range(n - 1):
        operators.append(I)
    operators.append(X)
    operators_list.append(operators)
    return qubits_list, operators_list


def control_operators_rotation(backend, qubits, mode='L'):
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    qubits_list, operators_list = operators_rotation(backend, qubits[1:], mode)

    for i in range(len(qubits_list)):
        qubits_list[i] = [qubits[0]] + qubits_list[i]
        operators_list[i] = [E2] + operators_list[i]

    qubits_list = [[qubits[0]]] + qubits_list
    operators_list = [[E1]] + operators_list
    return qubits_list, operators_list


@_register
@lru_cache(maxsize=None)
def L(backend, qubits):
    """
    L gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    qubits_list, operators_list = operators_rotation(backend, qubits, mode='L')
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def CL(backend, qubits):
    """
    CL gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    qubits_list, operators_list = control_operators_rotation(backend,
                                                             qubits,
                                                             mode='L')
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def R(backend, qubits):
    """
    L gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    qubits_list, operators_list = operators_rotation(backend, qubits, mode='R')
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def CR(backend, qubits):
    """
    CR gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    qubits_list, operators_list = control_operators_rotation(backend,
                                                             qubits,
                                                             mode='R')
    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def Kb(backend, qubits, *theta_list):
    """
    Kb gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    assert len(qubits) >= 2
    assert len(theta_list) + 1 == len(qubits)
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    H = backend.astensor(tensors.H())
    X = backend.astensor(tensors.X())

    n = len(qubits)
    qubits_list = [qubits for _ in range(n)]
    operators_list = []
    for i in range(n - 1):
        operators = []
        for j in range(i):
            operators.append(backend.astensor(tensors.Ry(theta_list[j])) @ E1)
        operators.append(backend.astensor(tensors.Ry(theta_list[i])) @ E2)
        for j in range(i + 1, n):
            operators.append(H)
        operators_list.append(operators)

    operators = []
    for j in range(n - 1):
        operators.append(backend.astensor(tensors.Ry(theta_list[j])) @ E1)
    operators.append(X)
    operators_list.append(operators)

    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def Kbinv(backend, qubits, *theta_list):
    """
    Kb gate used in the quantum walk.
        Ref: https://arxiv.org/pdf/1609.00173.pdf
    """
    assert len(qubits) >= 2
    assert len(theta_list) + 1 == len(qubits)
    E1 = backend.astensor(tensors.E1())
    E2 = backend.astensor(tensors.E2())
    H = backend.astensor(tensors.H())
    X = backend.astensor(tensors.X())

    n = len(qubits)
    qubits_list = [qubits for _ in range(n)]
    operators_list = []
    for i in range(n - 1):
        operators = []
        for j in range(i):
            operators.append(
                E1 @ backend.astensor(tensors.Ryinv(theta_list[j])))
        operators.append(E2 @ backend.astensor(tensors.Ryinv(theta_list[i])))
        for j in range(i + 1, n):
            operators.append(H)
        operators_list.append(operators)

    operators = []
    for j in range(n - 1):
        operators.append(E1 @ backend.astensor(tensors.Ryinv(theta_list[j])))
    operators.append(X)
    operators_list.append(operators)

    return MultiRankGate(qubits_list, operators_list)


@_register
@lru_cache(maxsize=None)
def U0(backend, qubits):
    assert len(qubits) >= 2
    marked_states = [tuple(0 for _ in range(len(qubits)))]
    return Uf(backend, qubits, *marked_states)
