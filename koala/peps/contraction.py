"""
This module implements contraction algorithms.
"""

from collections import namedtuple
import numpy as np
from tensorbackends.interface import ReducedSVD, ImplicitRandomizedSVD

from . import sites


Snake = namedtuple('Snake', [])
ABMPS = namedtuple('ABMPS', ['svd_option'])
BMPS = namedtuple('BMPS', ['svd_option'])
Square = namedtuple('Square', ['svd_option'])
TRG = namedtuple('TRG', ['svd_option'])


def contract(state, option):
    """
    Contract the PEPS to a single tensor or a scalar(a "0-tensor").

    Parameters
    ----------
    approach: str, optional
        The approach to contract.

    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.
    """
    if option is None:
        option = BMPS(None)
    if isinstance(option, Snake):
        return contract_snake(state)
    elif isinstance(option, ABMPS):
        return contract_ABMPS(state, svd_option=option.svd_option)
    elif isinstance(option, BMPS):
        return contract_BMPS(state, svd_option=option.svd_option)
    elif isinstance(option, Square):
        return contract_squares(state, svd_option=option.svd_option)
    elif isinstance(option, TRG):
        return contract_TRG(state, svd_option=option.svd_option)
    else:
        raise ValueError(f'unknown contraction option: {option}')


def contract_ABMPS(state, mps_mult_mpo=None, svd_option=None):
    """
    Contract the PEPS by performing alternating vertical and horizontal bondary contractions.

    Parameters
    ----------
    mps_mult_mpo: method or None, optional
        The method used to apply an MPS to another MPS/MPO.

    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.
    """
    horizontal = False
    while state.ncol > 2 and state.nrow > 2:
        edge = state[:,:2] if horizontal else state[:2]
        body = state[:,2:] if horizontal else state[2:]
        state = contract_to_MPS(edge, horizontal=horizontal, svd_option=svd_option).concatenate(body, int(horizontal))
    return contract_BMPS(state, mps_mult_mpo)


def contract_BMPS(state, mps_mult_mpo=None, svd_option=None):
    """
    Contract the PEPS by contracting each MPS layer.

    Parameters
    ----------
    mps_mult_mpo: method or None, optional
        The method used to apply an MPS to another MPS/MPO.

    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.
    """
    # contract boundary MPS down
    mps = contract_to_MPS(state, False, mps_mult_mpo, svd_option).grid.reshape(-1)
    # contract the last MPS to a single tensor
    result = mps[0]
    for tsr in mps[1:]:
        result = sites.contract_y(result, tsr)
    return result.item() if result.size == 1 else result.reshape(
        *[int(result.size ** (1 / state.grid.size))] * state.grid.size
        ).transpose(*[i + j * state.nrow for i, j in np.ndindex(*state.shape)])


def contract_env(state, row_range, col_range, svd_option=None):
    """
    Contract the surrounding environment to four MPS around the core sites.

    Parameters
    ----------
    row_range: tuple or int
        A two-int tuple specifying the row range of the core sites, i.e. [row_range[0] : row_range[1]].
        If only an int is given, it is equivalent to (row_range, row_range+1).

    col_range: tuple or int
        A two-int tuple specifying the column range of the core sites, i.e. [:, col_range[0] : col_range[1]].
        If only an int is given, it is equivalent to (col_range, col_range+1).

    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: PEPS
        The new PEPS consisting of core sites and contracted environment.
    """
    if isinstance(row_range, int):
        row_range = (row_range, row_range+1)
    if isinstance(col_range, int):
        col_range = (col_range, col_range+1)
    mid_peps = state[row_range[0]:row_range[1]].copy()
    if row_range[0] > 0:
        mid_peps = state[:row_range[0]].contract_to_MPS(svd_option=svd_option).concatenate(mid_peps)
    if row_range[1] < state.nrow:
        mid_peps = mid_peps.concatenate(state[row_range[1]:].contract_to_MPS(svd_option=svd_option))
    env_peps = mid_peps[:,col_range[0]:col_range[1]]
    if col_range[0] > 0:
        env_peps = mid_peps[:,:col_range[0]].contract_to_MPS(horizontal=True, svd_option=svd_option).concatenate(env_peps, axis=1)
    if col_range[1] < mid_peps.shape[1]:
        env_peps = env_peps.concatenate(mid_peps[:,col_range[1]:].contract_to_MPS(horizontal=True, svd_option=svd_option), axis=1)
    return env_peps


def contract_snake(state):
    """
    Contract the PEPS by contracting sites in the row-major order.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.

    References
    ----------
    https://arxiv.org/pdf/1905.08394.pdf
    """
    head = state.grid[0,0]
    for tsr in state.grid[0,1:]:
        head = sites.contract_y(head, tsr)
    for i, mps in enumerate(state.grid[1:]):
        head = head.transpose(2, 1, 0, 3, 4, 5)
        for tsr in mps[::2 * (i % 2) - 1]:
            head = state.backend.einsum('agbcdef->a(gb)cdef', 
                head.reshape(*((head.shape[0] // tsr.shape[0], tsr.shape[0]) + head.shape[1:])))
            tsr = state.backend.einsum('agbcdef->a' + ('bc(gd)ef' if i % 2 else 'dc(gb)ef'), tsr.reshape(*((1,) + tsr.shape)))
            head = sites.contract_y(head, tsr)
    return head.item() if head.size == 1 else head.reshape(*[int(head.size ** (1 / state.grid.size))] * state.grid.size)


def contract_squares(state, svd_option=None):
    """
    Contract the PEPS by contracting two neighboring tensors to one recursively.
    The neighboring relationship alternates from horizontal and vertical.

    Parameters
    ----------
    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.
    """
    from .peps import PEPS
    tn = state.grid
    new_tn = np.empty((int((state.nrow + 1) / 2), state.ncol), dtype=object)
    for ((i, j), a), b in zip(np.ndenumerate(tn[:-1:2,:]), tn[1::2,:].flat):
        new_tn[i,j] = sites.contract_x(a, b)
        if svd_option is not None:
            new_tn[i,j-1], new_tn[i,j] = sites.reduce_y(new_tn[i,j-1], new_tn[i,j], svd_option)
    # append the left edge if nrow/ncol is odd
    if state.nrow % 2 == 1:
        for i, a in enumerate(tn[-1]):
            new_tn[-1,i] = a.copy()
    # base case
    if new_tn.shape == (1, 1):
        return new_tn[0,0].item() if new_tn[0,0].size == 1 else new_tn[0,0]
    # alternate the neighboring relationship and contract recursively
    return contract_squares(PEPS(new_tn, state.backend).rotate(), svd_option)


def contract_to_MPS(state, horizontal=False, mps_mult_mpo=None, svd_option=None):
    """
    Contract the PEPS to an MPS.

    Parameters
    ----------
    horizontal: bool, optional
        Control whether to contract from top to bottom or from left to right. Will affect the output MPS direction.

    mps_mult_mpo: method or None, optional
        The method used to apply an MPS to another MPS/MPO.

    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: PEPS
        The resulting MPS (as a `PEPS` object of shape `(1, N)` or `(M, 1)`).
    """
    from .peps import PEPS
    if mps_mult_mpo is None:
        mps_mult_mpo = _mps_mult_mpo
    if horizontal:
        state.rotate(-1)
    mps = state.grid[0]
    for i, mpo in enumerate(state.grid[1:]):
        mps = mps_mult_mpo(mps, mpo, svd_option)
    mps = mps.reshape(1, -1)
    p = PEPS(mps, state.backend)
    return p.rotate() if horizontal else p


def contract_TRG(state, svd_option=None):
    """
    Contract the PEPS using Tensor Renormalization Group.

    Parameters
    ----------
    svdargs: dict, optional
        Arguments for SVD truncation. Will perform SVD if given.

    Returns
    -------
    output: state.backend.tensor or scalar
        The contraction result.

    References
    ----------
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.120601
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.78.205116
    """
    # base case
    if state.shape <= (2, 2):
        return state.contract_BMPS(svd_option)
    # if not svd_option:
    #     svd_option = {'rank': None}
    # SVD each tensor into two
    tn = np.empty(state.shape + (2,), dtype=object)
    for (i, j), tsr in np.ndenumerate(state.grid):
        tn[i,j,0], tn[i,j,1] = state.backend.einsvd('abcdpq->abi,icdpq' if (i+j) % 2 == 0 else 'abcdpq->aidpq,bci', tsr)
        tn[i,j,(i+j)%2] = tn[i,j,(i+j)%2].reshape(tn[i,j,(i+j)%2].shape + (1, 1))
    return _contract_TRG(state, tn, svd_option)


def _contract_TRG(state, tn, svd_option=None):
    from .peps import PEPS
    # base case
    if tn.shape == (2, 2, 2):
        p = np.empty((2, 2), dtype=object)
        for i, j in np.ndindex((2, 2)):
            p[i,j] = state.backend.einsum('abipq,icdPQ->abcd(pP)(qQ)' if (i+j) % 2 == 0 else 'aidpq,bciPQ->abcd(pP)(qQ)', tn[i,j][0], tn[i,j][1])
        return contract_BMPS(PEPS(p, state.backend))

    # contract specific horizontal and vertical bonds and SVD truncate the generated squared bonds
    for i, j in np.ndindex(tn.shape[:2]):
        if j > 0 and j % 2 == 0:
            k = 1 - i % 2
            l = j - ((i // 2 * 2 + j) % 4 == 0)
            tn[i,l][k] = state.backend.einsum('ibapq,ABiPQ->A(bB)a(pP)(qQ)' if k else 'biapq,BAiPQ->(bB)Aa(pP)(qQ)', tn[i,j-1][k], tn[i,j][k])
            if i % 2 == 1:
                # FIXME
                tn[i-1,l][1], tn[i,l][0] = state.backend.einsum('aidpq,iBCPQ->aBCd(pP)(qQ)', tn[i-1,l][1], tn[i,l][0], svd_option)
        if i > 0 and i % 2 == 0:
            k = 1 - j % 2
            l = int((i + j // 2 * 2) % 4 == 0)
            tn[i-l,j][l] = state.backend.einsum('biapq,iBAPQ->(bB)Aa(pP)(qQ)' if k else 'aibpq,iABPQ->aA(bB)(pP)(qQ)', tn[i-1,j][1], tn[i,j][0])
            if j % 2 == 1:
                # FIXME
                tn[i-l,j-1][l], tn[i-l,j][l] = state.backend.einsum('icdpq,ABiPQ->ABcd(pP)(qQ)', tn[i-l,j-1][l], tn[i-l,j][l], svd_option)

    # contract specific diagonal bonds and generate a smaller tensor network
    new_tn = np.empty((tn.shape[0] // 2 + 1, tn.shape[1] // 2 + 1, 2), dtype=object)
    for i, j in np.ndindex(tn.shape[:2]):
        m, n = (i + 1) // 2, (j + 1) // 2
        if (i + j) % 4 == 2 and i % 2 == 0:
            if tn[i,j][0] is None:
                new_tn[m,n][1] = tn[i,j][1]
            elif tn[i,j][1] is None:
                new_tn[m,n][1] = tn[i,j][0]
            else:
                new_tn[m,n][1] = state.backend.einsum('abipq,iCAPQ->bC(aA)(pP)(qQ)' if i == 0 else 'aibpq,iCBPQ->aCb+Bp+Pq+Q', tn[i,j][0], tn[i,j][1])
        elif (i + j) % 4 == 0 and i % 2 == 1:
            new_tn[m,n][0] = state.backend.einsum('abipq,iBCPQ->a(bB)C(pP)(qQ)', tn[i,j][0], tn[i,j][1])
        elif (i + j) % 4 == 3 and i % 2 == 0:
            new_tn[m,n][1] = state.backend.einsum('aibpq,ACiPQ->(aA)Cb(pP)(qQ)', tn[i,j][0], tn[i,j][1])
        elif (i + j) % 4 == 3 and i % 2 == 1:
            new_tn[m,n][0] = state.backend.einsum('aibpq,CBiPQ->aC(bB)(pP)(qQ)', tn[i,j][0], tn[i,j][1])
        else:
            if new_tn[m,n][0] is None:
                new_tn[m,n][0] = tn[i,j][0]
            if new_tn[m,n][1] is None:
                new_tn[m,n][1] = tn[i,j][1]

    # SVD truncate the squared bonds generated by the diagonal contractions
    for i, j in np.ndindex(new_tn.shape[:2]):
        if (i + j) % 2 == 0 and new_tn[i,j][0] is not None and new_tn[i,j][1] is not None:
            # FIXME
            new_tn[i,j][0], new_tn[i,j][1] = state.backend.einsum('abipq,iCDPQ->abCD(pP)(qQ)', new_tn[i,j][0], new_tn[i,j][1], svd_option)
        elif (i + j) % 2 == 1:
            # FIXME
            new_tn[i,j][0], new_tn[i,j][1] = state.backend.einsum('aidpq,BCiPQ->aBCd(pP)(qQ)', new_tn[i,j][0], new_tn[i,j][1], svd_option)

    return _contract_TRG(state, new_tn, svd_option)


def _mps_mult_mpo(mps, mpo, svd_option=None):
    # if mpo[0].shape[2] == 1:
    #     svd_option = None
    new_mps = np.empty_like(mps)
    for i, (s, o) in enumerate(zip(mps, mpo)):
        if isinstance(svd_option, ImplicitRandomizedSVD):
            if i == 0:
                new_mps[0] = s.backend.einsum('abidpq,iBcDPQ->abBc(dD)(pP)(qQ)', s, o)
            else:
                new_mps[i-1], new_mps[i] = s.backend.einsumsvd('aijcdpP,AbkiqQ,kBCjrR->aIcdpP,AbBCI(qr)(QR)', new_mps[i-1], s, o, option=svd_option)
                if i == len(mps)-1:
                    new_mps[-1] = s.backend.einsum('abBcdpq->a(bB)cdpq', new_mps[-1])
        else:
            new_mps[i] = sites.contract_x(s, o)
            if svd_option is not None and i > 0:
                new_mps[i-1], new_mps[i] = sites.reduce_y(new_mps[i-1], new_mps[i], svd_option)
    return new_mps


def create_env_cache(state1, state2, bmps_option):
    assert state1.shape == state2.shape
    peps_obj = state1.dagger().apply(state2)
    _up, _down = {}, {}
    for i in range(peps_obj.shape[0]):
        _up[i] = contract_to_MPS(peps_obj[:i], svd_option=bmps_option.svd_option) if i != 0 else None
        _down[i] = contract_to_MPS(peps_obj[i+1:], svd_option=bmps_option.svd_option) if i != state1.nrow - 1 else None
    return _up, _down


def inner_with_env(state, env, up_idx, down_idx, bmps_option):
    up, down = env[0][up_idx], env[1][down_idx]
    if up is None and down is None:
        peps_obj = state
    elif up is None:
        peps_obj = state.concatenate(down)
    elif down is None:
        peps_obj = up.concatenate(state)
    else:
        peps_obj = up.concatenate(state).concatenate(down)
    return peps_obj.contract(bmps_option)
