import math
import tensorbackends
from koala import candecomp
import numpy as np

from collections import namedtuple
import tracemalloc

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def generate_circuit(marked_states, nsite):
    num_layers = math.floor(math.pi / 4. * np.sqrt(2**(nsite)))
    # num_layers = math.floor(math.pi / 4. * np.sqrt(2**(nsite) / len(marked_states)))
    print(f"num_layers is {num_layers}")

    Hs_gate = Gate('Hs', [], [j for j in range(nsite)])
    flip_gate = Gate('FLIP', [], [0])
    U0_gate = Gate('U0', [], [j for j in range(nsite)])
    Uf_gate = Gate('Uf', marked_states, [j for j in range(nsite)])

    circuit = []
    circuit.append(Hs_gate)
    for _ in range(num_layers):
        circuit += [Uf_gate, Hs_gate, U0_gate, Hs_gate, flip_gate]
    return circuit


def grover_candecomp(marked_states,
                     backend='numpy',
                     rank_threshold=800,
                     compress_ratio=0.25,
                     cp_tol=1e-5,
                     cp_maxiter=60,
                     cp_inneriter=20,
                     num_als_init=1,
                     init_als='random',
                     mode='als',
                     debug=True):
    nsite = len(marked_states[0])
    circuit = generate_circuit(marked_states, nsite)
    qstate = candecomp.basis(nsite=nsite, backend=backend)
    if mode == "als":
        qstate.apply_circuit(circuit,
                             rank_threshold=rank_threshold,
                             compress_ratio=compress_ratio,
                             cp_tol=cp_tol,
                             cp_maxiter=cp_maxiter,
                             cp_inneriter=cp_inneriter,
                             num_als_init=num_als_init,
                             init_als=init_als,
                             use_prev_factor=True,
                             debug=debug)
    elif mode == "direct":
        qstate.apply_circuit_grover(circuit,
                                    rank_threshold=rank_threshold,
                                    debug=debug)
    return qstate


def get_factors_from_state(state, backend):
    shape = (1, 2)
    factors = []
    for bit in state:
        if bit == 0:
            factor = backend.astensor(np.array([1, 0]).reshape(shape))
        elif bit == 1:
            factor = backend.astensor(np.array([0, 1]).reshape(shape))
        factors.append(factor)
    return factors


if __name__ == '__main__':
    backend = 'numpy'
    nsite = 15
    num_marked_states = 20
    debug = True
    rank_threshold = 2
    compress_ratio = 0.5
    cp_tol = 1e-5
    cp_maxiter = 100
    cp_inneriter = 20
    init_als = 'random'
    num_als_init = 3
    mode = 'direct'

    tb = tensorbackends.get(backend)

    if num_marked_states == 1:
        marked_states = [tuple([1 for _ in range(nsite)])]
    else:
        marked_states = [
            tuple(np.random.randint(2, size=nsite))
            for _ in range(num_marked_states)
        ]
        marked_states[0] = tuple([1 for _ in range(nsite)])

    marked_states_factors = [
        get_factors_from_state(state, tb) for state in marked_states
    ]

    tracemalloc.start()

    qstate = grover_candecomp(marked_states,
                              backend=backend,
                              rank_threshold=rank_threshold,
                              compress_ratio=compress_ratio,
                              cp_tol=cp_tol,
                              cp_maxiter=cp_maxiter,
                              cp_inneriter=cp_inneriter,
                              num_als_init=num_als_init,
                              init_als=init_als,
                              mode=mode,
                              debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    overall_item_fidelity = 0.
    for i in range(num_marked_states):
        item_fidelity = candecomp.fidelity(marked_states_factors[i],
                                           qstate.factors, tb)**2
        print(f"Fidelity for {i} is {item_fidelity}")
        overall_item_fidelity += item_fidelity
    print(f"Overall item fidelity is {overall_item_fidelity}")
    print(f"Fidelity average is {qstate.fidelity_avg ** 2}")
