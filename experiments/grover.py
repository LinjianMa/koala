import math
import tensorbackends
from koala import candecomp
import numpy as np

from collections import namedtuple
import tracemalloc

Gate = namedtuple('Gate', ['name', 'parameters', 'qubits'])


def generate_circuit(marked_states, nsite):
    num_layers = math.floor(math.pi / 4. * np.sqrt(2**(nsite) / len(marked_states)))
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
                     init_als='random',
                     debug=True):
    nsite = len(marked_states[0])
    circuit = generate_circuit(marked_states, nsite)
    qstate = candecomp.basis(nsite=nsite, backend=backend)
    qstate.apply_circuit(circuit,
                         rank_threshold=rank_threshold,
                         compress_ratio=compress_ratio,
                         cp_tol=cp_tol,
                         cp_maxiter=cp_maxiter,
                         cp_inneriter=cp_inneriter,
                         init_als=init_als,
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
    nsite = 20
    num_marked_states = 4
    debug = True
    rank_threshold = 5
    compress_ratio = 0.5
    cp_tol = 1e-5
    cp_maxiter = 100
    cp_inneriter = 20
    init_als = 'random'

    tb = tensorbackends.get(backend)

    marked_states = [
        tuple(np.random.randint(2, size=nsite))
        for _ in range(num_marked_states)
    ]
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
                              init_als=init_als,
                              debug=debug)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'current_memory is {current_memory / (1024 * 1024)} MB')
    print(f'peak_memory is {peak_memory / (1024 * 1024)} MB')

    for i in range(num_marked_states):
        print(
            f"Fidelity for {i} is {candecomp.fidelity(marked_states_factors[i], qstate.factors, tb)}"
        )
