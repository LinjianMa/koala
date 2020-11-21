import math
import tensorbackends
import copy
from koala import candecomp
import numpy as np

from collections import namedtuple
from grover import get_factors_from_state
import tracemalloc

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def generate_walk_operator(nsite, mode="complete"):
    assert nsite % 2 == 0
    nsite_vertices = nsite // 2

    second_register = [n for n in range(nsite_vertices, 2 * nsite_vertices)]
    U0_gate = Gate('U0', [], second_register)

    circuit = []

    if mode == "complete":
        thetas = []
        for i in range(nsite_vertices - 1):
            frac = (2**(nsite_vertices -
                        (i + 1)) - 1) / (2**(nsite_vertices - i) - 1)
            theta = np.arccos(np.sqrt(frac))
            thetas.append(theta)

        for i in range(nsite_vertices - 1):
            c_qubit = nsite_vertices - i - 1
            cd_qubits = [
                j for j in range(nsite_vertices, 2 * nsite_vertices - i)
            ]
            qubits = [c_qubit] + cd_qubits
            L_gate = Gate('CL', [], qubits)
            circuit.append(L_gate)
        circuit.append(Gate('CX', [], [0, nsite_vertices]))

        Kbinv_gate = Gate('Kbinv', thetas, second_register)
        Kb_gate = Gate('Kb', thetas, second_register)
        circuit += [Kbinv_gate, U0_gate, Kb_gate]

        for i in range(nsite_vertices - 1):
            c_qubit = nsite_vertices - i - 1
            cd_qubits = [
                j for j in range(nsite_vertices, 2 * nsite_vertices - i)
            ]
            qubits = [c_qubit] + cd_qubits
            L_gate = Gate('CR', [], qubits)
            circuit.append(L_gate)
        circuit.append(Gate('CX', [], [0, nsite_vertices]))

    elif mode == "loop":
        Hs_gate = Gate('Hs', [], second_register)
        circuit += [Hs_gate, U0_gate, Hs_gate]

    for i in range(nsite_vertices):
        circuit.append(Gate('SWAP', [], [i, nsite_vertices + i]))

    return circuit


def generate_circuit(nsite, mode="loop"):
    walk_step = generate_walk_operator(nsite, mode=mode)

    marked_states = [tuple([1 for _ in range(nsite // 2)])]
    Uf_gate = Gate('Uf', marked_states, [j for j in range(nsite // 2)])

    circuit = []
    num_layers = math.floor(math.pi / 4. * np.sqrt(2**(nsite // 2)))
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
                    mode="loop",
                    cpdmode="als",
                    debug=True):
    circuit = generate_circuit(nsite, mode)
    qstate = candecomp.uniform(nsite=nsite, backend=backend)

    if cpdmode == "als":
        qstate.apply_circuit(circuit,
                             rank_threshold=rank_threshold,
                             hard_compression=True,
                             cp_tol=cp_tol,
                             cp_maxiter=cp_maxiter,
                             cp_inneriter=cp_inneriter,
                             init_als=init_als,
                             num_als_init=num_als_init,
                             use_prev_factor=True,
                             debug=debug)
    elif cpdmode == "direct":
        qstate.apply_circuit_grover(circuit,
                                    rank_threshold=rank_threshold,
                                    debug=debug)
    return qstate


def build_marked_states(nsite_vertices, mode):
    if mode == "loop":
        num_edges = 2**nsite_vertices
    elif mode == "complete":
        num_edges = 2**nsite_vertices - 1

    marked_vertex_state = [1 for _ in range(nsite_vertices)]
    marked_states = []

    for num in range(num_edges):
        list_binary_str = list(format(num, "b"))
        length = len(list_binary_str)
        marked_edge_state = copy.deepcopy(marked_vertex_state)

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
    nsite = 22
    debug = True
    mode = "loop"
    cpdmode = "direct"

    # ALS arguments
    rank_threshold = 2
    cp_tol = 1e-8
    cp_maxiter = 100
    cp_inneriter = 20
    num_als_init = 3
    init_als = 'random'

    assert nsite % 2 == 0
    nsite_vertices = nsite // 2

    tb = tensorbackends.get(backend)

    marked_states = build_marked_states(nsite_vertices, mode)
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
                             mode=mode,
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
