import math
import tensorbackends
from koala import candecomp
import numpy as np

from collections import namedtuple
from scipy import fft
import tracemalloc

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


def qft_candecomp(qstate,
                  rank_threshold=800,
                  compress_ratio=0.25,
                  cp_tol=1e-5,
                  cp_maxiter=60,
                  cp_inneriter=20,
                  init_als='random',
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
                         debug=debug)


def fidelity(out_vector, true_vector):
    return out_vector @ true_vector.conj() / (tb.norm(true_vector) *
                                              tb.norm(out_vector))


def relative_residual(out_vector, true_vector):
    return tb.norm(out_vector - true_vector) / tb.norm(true_vector)


def argsort_diff(out_vector, true_vector):
    out_vector_argsort = tb.argsort(tb.absolute(out_vector))
    true_vector_argsort = tb.argsort(tb.absolute(true_vector))
    argsort_diff = out_vector_argsort - true_vector_argsort
    return tb.count_nonzero(argsort_diff)


if __name__ == '__main__':
    backend = 'numpy'
    nsite = 28  # statevector maximum 14
    debug = True
    rank_threshold = 400
    compress_ratio = 0.5
    cp_tol = 1e-5
    cp_maxiter = 100
    cp_inneriter = 20
    in_state = 'random'
    init_als = 'random'

    # backend = 'numpy'
    # nsite = 24
    # debug = True
    # rank_threshold = 10
    # compress_ratio = 0.5
    # cp_tol = 1e-5
    # cp_maxiter = 100
    # cp_inneriter = 20
    # in_state = 'rectangular_pulse'
    # init_als = 'factors'

    tb = tensorbackends.get(backend)

    if in_state == 'random':
        qstate = candecomp.random(nsite=nsite, rank=1, backend=backend)
    elif in_state == 'rectangular_pulse':
        qstate = candecomp.rectangular_pulse(nsite=nsite, backend=backend)

    statevector = qstate.get_statevector()

    if backend == 'numpy':
        out_true = tb.astensor(fft(statevector.ravel(), norm="ortho"))
    elif backend == 'ctf':
        out_true = tb.astensor(
            fft(statevector.ravel().to_nparray(), norm="ortho"))

    tracemalloc.start()

    qft_candecomp(qstate,
                  rank_threshold=rank_threshold,
                  compress_ratio=compress_ratio,
                  cp_tol=cp_tol,
                  cp_maxiter=cp_maxiter,
                  cp_inneriter=cp_inneriter,
                  init_als=init_als,
                  debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    out_statevector = qstate.get_statevector().ravel()

    print(
        f"Relative residual norm is {relative_residual(out_statevector, out_true)}"
    )
    print(f"Fidelity is {fidelity(out_statevector, out_true)}")
    print(f"Fidelity lower bound is {qstate.fidelity_lower}")
    print(f"Fidelity average is {qstate.fidelity_avg}")
