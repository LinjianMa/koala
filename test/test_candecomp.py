import unittest
import copy
import tensorbackends
import ctf
import numpy as np

from scipy import fft
from tensorbackends.utils import test_with_backend
from koala import Observable, candecomp, Gate, tensors
from experiments.qft import qft_candecomp


@test_with_backend()
class CanonicalDecomp(unittest.TestCase):
    def test_apply_rank_one_gate(self, backend):
        qstate = candecomp.random(nsite=4, rank=5, backend=backend)
        init_factors = copy.deepcopy(qstate.factors)
        qstate.apply_circuit([Gate('H', [], [0])])
        diff = qstate.factors[0] - init_factors[0] @ backend.astensor(
            tensors.H())
        self.assertTrue(
            np.isclose(backend.einsum("ab,ab->", diff, diff.conj()), 0.))

    def test_qft_with_full_rank(self, backend):
        nsite = 8  # maximum 14
        debug = False
        tb = tensorbackends.get(backend)

        qstate = candecomp.random(nsite=nsite, rank=1, backend=backend)
        statevector = qstate.get_statevector()

        qft_candecomp(qstate, debug=debug)
        out_statevector = qstate.get_statevector()

        if isinstance(statevector.unwrap(), np.ndarray):
            out_true = tb.astensor(fft(statevector.ravel(), norm="ortho"))
        elif isinstance(statevector.unwrap(), ctf.core.tensor):
            out_true = tb.astensor(
                fft(statevector.ravel().to_nparray(), norm="ortho"))

        self.assertTrue(
            np.isclose(tb.norm(out_statevector.ravel() - out_true), 0.))
