import math
import tensorbackends
import copy
from koala import candecomp
import numpy as np

from collections import namedtuple
from grover import get_factors_from_state
import tracemalloc

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def generate_walk_operator(nsite):
    assert nsite % 2 == 0
    nsite_vertices = nsite // 2 - 1

    gates_H1 = [0, 1] + ['H' for _ in range(nsite_vertices)]
    gates_H2 = [1, 0] + ['H' for _ in range(nsite_vertices)]
    gates_Z1 = [0, 'Z'] + [0 for _ in range(nsite_vertices)]
    gates_Z2 = [1, 'Z'] + [0 for _ in range(nsite_vertices)]
    sites = [0, nsite_vertices + 1
             ] + [nsite_vertices + 2 + i for i in range(nsite_vertices)]

    circuit = []
    circuit.append(Gate('General_control', gates_H1, sites))
    circuit.append(Gate('General_control', gates_Z1, sites))
    circuit.append(Gate('General_control', gates_H1, sites))

    circuit.append(Gate('General_control', gates_H2, sites))
    circuit.append(Gate('CX', [], [0, nsite_vertices + 1]))
    circuit.append(Gate('General_control', gates_Z2, sites))
    circuit.append(Gate('CX', [], [0, nsite_vertices + 1]))
    circuit.append(Gate('General_control', gates_H2, sites))

    for i in range(nsite // 2):
        circuit.append(Gate('SWAP', [], [i, nsite // 2 + i]))

    return circuit


def generate_circuit(nsite):
    walk_step = generate_walk_operator(nsite)

    marked_states = [tuple([1 for _ in range(nsite // 2 - 1)])]
    Uf_gate = Gate('Uf', marked_states, [j for j in range(1, nsite // 2)])

    circuit = []
    num_layers = math.floor(math.pi / 4. * np.sqrt(2**(nsite // 2 - 1)))
    print("num_layers is: ", num_layers)
    for i in range(num_layers):
        circuit += walk_step
        circuit += walk_step
        circuit.append(Uf_gate)

    return circuit


def qwalk_candecomp(nsite,
                    backend='numpy',
                    rank_threshold=800,
                    cp_tol=1e-5,
                    cp_maxiter=60,
                    cp_inneriter=20,
                    init_als='random',
                    num_als_init=5,
                    cpdmode="als",
                    debug=True):
    circuit = generate_circuit(nsite)
    qstate = candecomp.bipartite_uniform(nsite=nsite, backend=backend)

    if cpdmode == "als":
        qstate.apply_circuit(circuit,
                             rank_threshold=rank_threshold,
                             hard_compression=True,
                             cp_tol=cp_tol,
                             cp_maxiter=cp_maxiter,
                             cp_inneriter=cp_inneriter,
                             init_als=init_als,
                             num_als_init=num_als_init,
                             use_prev_factor=False,
                             debug=debug)
    elif cpdmode == "direct":
        qstate.apply_circuit_direct_cpd(circuit,
                                        rank_threshold=rank_threshold,
                                        debug=debug)
    return qstate


def build_marked_states(nsite_vertices):
    marked_vertex_state = [1 for _ in range(nsite_vertices)]
    marked_states = []

    for num in range(2**nsite_vertices):
        list_binary_str = list(format(num, "b"))
        length = len(list_binary_str)
        marked_edge_state = [0]
        marked_edge_state += copy.deepcopy(marked_vertex_state)
        marked_edge_state += [1]

        for i in range(nsite_vertices - length):
            marked_edge_state.append(0)
        for i in range(length):
            if list_binary_str[i] == '0':
                marked_edge_state.append(0)
            else:
                marked_edge_state.append(1)

        marked_states.append(tuple(marked_edge_state))
    return marked_states


if __name__ == '__main__':
    backend = 'numpy'
    nsite = 24  # needs to be 8, 12, 16, 20.. to achieve high amplitude
    debug = True
    cpdmode = "als"

    # ALS arguments
    rank_threshold = 40
    cp_tol = 1e-8
    cp_maxiter = 100
    cp_inneriter = 20
    num_als_init = 3
    init_als = 'random'

    assert nsite % 2 == 0
    nsite_vertices = nsite // 2 - 1

    tb = tensorbackends.get(backend)

    marked_states = build_marked_states(nsite_vertices)
    print(marked_states)

    marked_states_factors = [
        get_factors_from_state(state, tb) for state in marked_states
    ]

    tracemalloc.start()

    qstate = qwalk_candecomp(nsite,
                             backend=backend,
                             rank_threshold=rank_threshold,
                             cp_tol=cp_tol,
                             cp_maxiter=cp_maxiter,
                             cp_inneriter=cp_inneriter,
                             init_als=init_als,
                             num_als_init=num_als_init,
                             cpdmode=cpdmode,
                             debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    overall_edge_fidelity = 0.
    for i in range(len(marked_states)):
        edge_fidelity = candecomp.fidelity(marked_states_factors[i],
                                           qstate.factors, tb)**2
        print(f"Fidelity for {i} is {edge_fidelity}")
        overall_edge_fidelity += edge_fidelity

    print(f"overall_edge_fidelity is {overall_edge_fidelity}")
    print(f"Fidelity average is {qstate.fidelity_avg ** 2}")
