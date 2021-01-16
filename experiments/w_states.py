import math
import tensorbackends
from koala import candecomp
import numpy as np


def W_state(num_products):
    tb = tensorbackends.get('numpy')
    if num_products == 1:
        factors = [None for _ in range(3)]
        factors[0] = tb.asarray([[1. + 0j, 0.], [1., 0.], [0., 1.]])
        factors[1] = tb.asarray([[1. + 0j, 0.], [0., 1.], [1., 0.]])
        factors[2] = tb.asarray([[0. + 0j, 1.], [1., 0.], [1., 0.]])
        return factors

    factors = W_state(num_products=num_products - 1)
    factors_w = W_state(num_products=1)

    for i in range(len(factors)):
        factors[i] = tb.vstack((factors[i], factors[i], factors[i]))

    for i in range(3):
        v1 = tb.vstack(
            tuple([factors_w[i][0] for _ in range(3**(num_products - 1))]))
        v2 = tb.vstack(
            tuple([factors_w[i][1] for _ in range(3**(num_products - 1))]))
        v3 = tb.vstack(
            tuple([factors_w[i][2] for _ in range(3**(num_products - 1))]))
        factors.append(tb.vstack((v1, v2, v3)))
    return factors


def als_w():
    tb = tensorbackends.get('numpy')
    factors = W_state(4)

    for factor in factors:
        print(factor)

    t1 = candecomp.CanonicalDecomp(factors, 'numpy').get_statevector()

    out_factors, _ = candecomp.als.als(factors,
                                       tb,
                                       15,
                                       tol=1e-14,
                                       max_iter=20000,
                                       inner_iter=20,
                                       init_als='random',
                                       num_als_init=100,
                                       debug=True)

    for factor in out_factors:
        print(factor)

    t2 = candecomp.CanonicalDecomp(out_factors, 'numpy').get_statevector()
    print(tb.norm(t1 - t2))


if __name__ == '__main__':
    als_w()
