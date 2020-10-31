"""
This module defines some common tensors in NumPy.
"""

from math import sqrt
from itertools import repeat

import numpy as np


def control(nctrl, tensor):
    nqubit = nctrl + tensor.ndim // 2
    result = np.eye(2**nqubit, dtype=complex)
    result[2**nctrl:, 2**nctrl:] = tensor
    return result.reshape(*repeat(2, nqubit * 2))


def I():
    return np.array([1, 0, 0, 1], dtype=complex).reshape(2, 2)


def H():
    return (np.array([1, 1, 1, -1], dtype=complex) / np.sqrt(2)).reshape(2, 2)


def X():
    return np.array([0, 1, 1, 0], dtype=complex).reshape(2, 2)


def Y():
    return np.array([0, -1j, 1j, 0], dtype=complex).reshape(2, 2)


def Z():
    return np.array([1, 0, 0, -1], dtype=complex).reshape(2, 2)


def S():
    return np.array([1, 0, 0, 1j], dtype=complex).reshape(2, 2)


def Sdag():
    return np.array([1, 0, 0, -1j], dtype=complex).reshape(2, 2)


def T():
    return U1(np.pi / 4)


def Tdag():
    return U1(-np.pi / 4)


def W():
    return (X() + Y()) / sqrt(2)


def sqrtX():
    return np.array([1 + 1j, 1 - 1j, 1 - 1j, 1 + 1j], dtype=complex).reshape(
        2, 2) / 2


def sqrtY():
    return np.array([1 + 1j, -1 - 1j, 1 + 1j, 1 + 1j], dtype=complex).reshape(
        2, 2) / 2


def sqrtZ():
    return S()


def sqrtW():
    return np.array([1 + 1j, -sqrt(2) * 1j,
                     sqrt(2), 1 + 1j], dtype=complex).reshape(2, 2) / 2


def R(theta):
    return U1(theta)

def Rinv(theta):
    return np.linalg.inv(R(theta))


def Ry(theta):
    # Note: this is the transpose of Ry comapred to the paper.
    # The reason is that the framework implements x^TG^T rather than Gx
    return np.array(
        [np.cos(theta),
         np.sin(theta), -np.sin(theta),
         np.cos(theta)],
        dtype=complex).reshape(2, 2)


def Ryinv(theta):
    return np.array(
        [np.cos(theta), -np.sin(theta),
         np.sin(theta),
         np.cos(theta)],
        dtype=complex).reshape(2, 2)


def U1(lmbda):
    return U3(0, 0, lmbda)


def U2(phi, lmbda):
    return U3(np.pi / 2, phi, lmbda)


def U3(theta, phi, lmbda):
    c, s = np.cos(theta), np.sin(theta)
    e_phi, e_lmbda = np.exp(1j * phi), np.exp(1j * lmbda)
    return np.array([c, -e_lmbda * s, e_phi * s, e_lmbda * e_phi * c],
                    dtype=complex).reshape(2, 2)


def SWAP():
    return np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    dtype=complex).reshape(2, 2, 2, 2)


def ISWAP():
    return np.array([1, 0, 0, 0, 0, 0, 1j, 0, 0, 1j, 0, 0, 0, 0, 0, 1],
                    dtype=complex).reshape(2, 2, 2, 2)


def XX():
    return np.einsum('ij,kl->ikjl', X(), X())


def XY():
    return np.einsum('ij,kl->ikjl', X(), Y())


def XZ():
    return np.einsum('ij,kl->ikjl', X(), Z())


def YY():
    return np.einsum('ij,kl->ikjl', Y(), Y())


def YZ():
    return np.einsum('ij,kl->ikjl', Y(), Z())


def ZZ():
    return np.einsum('ij,kl->ikjl', Z(), Z())


################# following are tensors used in the CP basis.
def E1():
    return np.array([1, 0, 0, 0], dtype=complex).reshape(2, 2)


def E2():
    return np.array([0, 0, 0, 1], dtype=complex).reshape(2, 2)
