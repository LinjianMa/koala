import math
import tensorbackends
from koala import candecomp
import numpy as np

from collections import namedtuple
from scipy import fft

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def theta(n):
    return -2 * math.pi / (2**n)


def generate_circuit(nsite):
    circuit = []

    for i in range(nsite - 1):
        circuit.append(Gate('H', [], [i]))
        theta_list = [theta(j) for j in range(2, nsite + 1 - i)]
        qubits = [j for j in range(i, nsite)]
        circuit.append(Gate('CRs', theta_list, qubits))
    circuit.append(Gate('H', [], [nsite - 1]))

    # swap
    for i in range(int(nsite / 2)):
        circuit.append(Gate('SWAP', [], [i, nsite - 1 - i]))

    return circuit


def qft_candecomp(qstate, debug):
    nsite = qstate.nsite
    circuit = generate_circuit(nsite)
    qstate.apply_circuit(circuit, debug=debug)


if __name__ == '__main__':
    backend = 'numpy'
    nsite = 8  # maximum 14
    debug = True
    tb = tensorbackends.get(backend)

    qstate = candecomp.random(nsite=nsite, rank=1, backend=backend)
    statevector = qstate.get_statevector()

    qft_candecomp(qstate, debug=debug)
    out_statevector = qstate.get_statevector()
    out_true = tb.astensor(fft(statevector.ravel(), norm="ortho"))

    print(tb.norm(out_statevector.ravel() - out_true))
