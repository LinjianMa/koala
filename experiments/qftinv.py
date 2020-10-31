import math
import tensorbackends
from koala import candecomp
import numpy as np

from collections import namedtuple
from scipy import fft
import tracemalloc

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def thetafunc(n):
    return -2 * math.pi / (2**n)


def generate_circuit(nsite):
    circuit = []

    # # swap
    # for i in reversed(range(int(nsite / 2))):
    #     circuit.append(Gate('SWAP', [], [i, nsite - 1 - i]))

    circuit.append(Gate('H', [], [nsite - 1]))

    for i in reversed(range(nsite - 1)):
        theta_list = [thetafunc(j) for j in range(2, nsite + 1 - i)]
        qubits = [j for j in range(i, nsite)]
        circuit.append(Gate('CRs_inv', theta_list, qubits))
        circuit.append(Gate('H', [], [i]))

    return circuit


def qftinv_candecomp(qstate,
                     rank_threshold=800,
                     compress_ratio=0.25,
                     cp_tol=1e-5,
                     cp_maxiter=60,
                     cp_inneriter=20,
                     init_als='random',
                     num_als_init=1,
                     debug=True):
    nsite = qstate.nsite
    circuit = generate_circuit(nsite)
    qstate.apply_circuit(circuit,
                         rank_threshold=rank_threshold,
                         compress_ratio=compress_ratio,
                         cp_tol=cp_tol,
                         cp_maxiter=cp_maxiter,
                         cp_inneriter=cp_inneriter,
                         init_als=init_als,
                         num_als_init=num_als_init,
                         debug=debug)


if __name__ == '__main__':
    backend = 'numpy'
    nsite = 28  # statevector maximum 14
    debug = True
    rank_threshold = 20
    compress_ratio = 0.5
    cp_tol = 1e-5
    cp_maxiter = 100
    cp_inneriter = 20
    init_als = 'mixed'
    theta = 0.5 + 0.5 / 2**nsite
    print(f"theta is {theta}")
    num_als_init = 3

    tb = tensorbackends.get(backend)

    qstate = candecomp.qft_inv_input(nsite=nsite, theta=theta, backend='numpy')

    # statevector = qstate.get_statevector()

    tracemalloc.start()

    qftinv_candecomp(qstate,
                     rank_threshold=rank_threshold,
                     compress_ratio=compress_ratio,
                     cp_tol=cp_tol,
                     cp_maxiter=cp_maxiter,
                     cp_inneriter=cp_inneriter,
                     init_als=init_als,
                     num_als_init=num_als_init,
                     debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    # out_statevector = qstate.get_statevector().ravel()
    # print(out_statevector)
    # vec1 = out_statevector[:2**(nsite-1)-10]
    # vec2 = out_statevector[2**(nsite-1)+10:]
    # print(out_statevector.conj()@out_statevector)
    # print(vec1.conj()@vec1 + vec2.conj()@vec2)

    print(f"Fidelity lower bound is {qstate.fidelity_lower}")
    print(f"Fidelity average is {qstate.fidelity_avg**2}")
