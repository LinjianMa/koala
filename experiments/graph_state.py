import math
import tensorbackends
import numpy as np
import tracemalloc

from koala import candecomp
from collections import namedtuple
from scipy import fft

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def generate_circuit(nsite, edges):
    circuit = []
    for edge in edges:
        circuit.append(Gate('CZ', [], edge))
    return circuit


def graph_candecomp(qstate,
                    edges,
                    alsmode='als',
                    rank_threshold=800,
                    cp_tol=1e-5,
                    cp_maxiter=60,
                    cp_inneriter=20,
                    init_als='random',
                    debug=True):
    nsite = qstate.nsite
    circuit = generate_circuit(nsite, edges)
    if alsmode == "als":
        qstate.apply_circuit(circuit,
                             rank_threshold=rank_threshold,
                             cp_tol=cp_tol,
                             cp_maxiter=cp_maxiter,
                             cp_inneriter=cp_inneriter,
                             init_als=init_als,
                             num_als_init=100,
                             debug=debug)
    elif alsmode == "direct":
        qstate.apply_circuit_direct_cpd(circuit,
                                        rank_threshold=rank_threshold,
                                        debug=debug)


def fidelity(out_vector, true_vector):
    return out_vector @ true_vector.conj() / (tb.norm(true_vector) *
                                              tb.norm(out_vector))


def relative_residual(out_vector, true_vector):
    return tb.norm(out_vector - true_vector) / tb.norm(true_vector)


def get_7_nodes_conjecture_state(backend):
    tb = tensorbackends.get(backend)

    zero = tb.astensor(np.asarray([1., 0.]))
    one = tb.astensor(np.asarray([0., 1.]))
    plus = tb.astensor(1. / np.sqrt(2) * np.asarray([1., 1.]))
    minus = tb.astensor(1. / np.sqrt(2) * np.asarray([1., -1.]))

    out1 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", minus, one, minus, zero, plus, zero, plus)
    out2 = -1. / (2) * tb.einsum("a,b,c,d,e,f,g->abcdefg", one, minus, zero,
                                 plus, zero, minus, one)
    out3 = -1. / (2) * tb.einsum("a,b,c,d,e,f,g->abcdefg", one, plus, one,
                                 plus, one, plus, one)
    out4 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", plus, zero, minus, one, minus, zero, plus)
    out5 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", plus, zero, minus, one, plus, one, minus)
    out6 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", plus, zero, plus, zero, minus, one, minus)
    out7 = -1. / (2) * tb.einsum("a,b,c,d,e,f,g->abcdefg", one, plus, one,
                                 minus, zero, minus, one)
    out8 = 1. / (2 * np.sqrt(2)) * tb.einsum("a,b,c,d,e,f,g->abcdefg", minus,
                                             one, plus, one, minus, zero, plus)
    out9 = -1. / (2) * tb.einsum("a,b,c,d,e,f,g->abcdefg", one, minus, zero,
                                 minus, one, plus, one)
    out10 = 1. / (2 * np.sqrt(2)) * tb.einsum("a,b,c,d,e,f,g->abcdefg", minus,
                                              one, plus, one, plus, one, minus)
    out11 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", plus, zero, plus, zero, plus, zero, plus)
    out12 = 1. / (2 * np.sqrt(2)) * tb.einsum(
        "a,b,c,d,e,f,g->abcdefg", minus, one, minus, zero, minus, one, minus)

    out = out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8 + out9 + out10 + out11 + out12
    return out.ravel()


def build_edges(nvertices, mode):
    edges = []
    if mode == 'ring':
        v = 0
        for v in range(nvertices // 2):
            edges.append([v * 2 + 1, v * 2])
            edges.append([v * 2 + 1, (v * 2 + 2) % nvertices])
        if nvertices % 2 != 0:
            edges.append([0, nvertices - 1])
    elif mode == 'line':
        for v in range(nvertices - 1):
            edges.append([v, v + 1])
    elif mode == 'double-line':
        for v in range(nvertices // 2 - 1):
            edges.append([v, v + 1])
        for v in range(nvertices // 2, nvertices - 1):
            edges.append([v, v + 1])
    return edges


if __name__ == '__main__':
    backend = 'numpy'
    nvertices = 8
    debug = True
    rank_threshold = 16
    cp_tol = 1e-10
    cp_maxiter = 3000
    cp_inneriter = 20
    init_als = 'random'
    mode = 'double-line'
    alsmode = 'als'

    tb = tensorbackends.get(backend)

    # define qstate
    qstate1 = candecomp.uniform(nvertices, backend=backend)
    qstate2 = candecomp.uniform(nvertices, backend=backend)
    statevector = qstate1.get_statevector()
    print(statevector)

    edges = build_edges(nvertices, mode)

    tracemalloc.start()

    graph_candecomp(qstate1,
                    edges,
                    alsmode=alsmode,
                    rank_threshold=rank_threshold,
                    cp_tol=cp_tol,
                    cp_maxiter=cp_maxiter,
                    cp_inneriter=cp_inneriter,
                    init_als=init_als,
                    debug=debug)
    graph_candecomp(qstate2,
                    edges,
                    rank_threshold=1e10,
                    cp_tol=cp_tol,
                    cp_maxiter=cp_maxiter,
                    cp_inneriter=cp_inneriter,
                    init_als=init_als,
                    debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    out_statevector = qstate1.get_statevector().ravel()
    out_true = qstate2.get_statevector().ravel()
    # out_conjecture = get_7_nodes_conjecture_state(backend)

    print(out_statevector)
    print(out_true)
    # print(tb.norm(out_true - out_conjecture))
    print(qstate1.factors)

    print(
        f"Relative residual norm is {relative_residual(out_statevector, out_true)}"
    )
    print(f"Fidelity is {fidelity(out_statevector, out_true)}")
    print(f"Fidelity lower bound is {qstate1.fidelity_lower}")
    print(f"Fidelity average is {qstate1.fidelity_avg}")
