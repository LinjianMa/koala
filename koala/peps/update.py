from tensorbackends.interface import ReducedSVD
import numpy as np
import scipy.linalg as la
import tensorbackends


class UpdateOption:
    def __str__(self):
        return '{}({})'.format(
            type(self).__name__,
            ','.join('{}={}'.format(k, v) for k, v in vars(self).items())
        )

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return type(self).__name__


class DirectUpdate(UpdateOption):
    def __init__(self, svd_option=None):
        self.svd_option = svd_option


class QRUpdate(UpdateOption):
    def __init__(self, svd_option=None):
        self.svd_option = svd_option


class DefaultUpdate(UpdateOption):
    def __init__(self, rank=None):
        self.rank = rank

class Timer:
    def __init__(self, backend, name):
        backend = tensorbackends.get(backend)
        if backend.name in {'ctf', 'ctfview'}:
            import ctf
            self.timer = ctf.timer(name)
        else:
            self.timer = None

    def __enter__(self):
        if self.timer is not None:
            self.timer.start()

    def __exit__(self, type, value, traceback):
        if self.timer is not None:
            self.timer.stop()

def apply_single_site_operator(state, operator, position):
    state.grid[position] = state.backend.einsum('ijklxp,xy->ijklyp', state.grid[position], operator)


def apply_local_pair_operator(state, operator, positions, update_option):
    if update_option is None:
        update_option = DefaultUpdate()
    if isinstance(update_option, DefaultUpdate):
        return apply_local_pair_operator_gram_qr_local(state, operator, positions, update_option.rank)
    elif isinstance(update_option, DirectUpdate):
        return apply_local_pair_operator_direct(state, operator, positions, update_option.svd_option)
    elif isinstance(update_option, QRUpdate):
        return apply_local_pair_operator_qr(state, operator, positions, update_option.svd_option)
    else:
        raise ValueError(f'unknown update option: {update_option}')


def apply_local_pair_operator_direct(state, operator, positions, svd_option):
    assert len(positions) == 2
    if svd_option is None:
        svd_option = ReducedSVD()
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        prod_subscripts = 'abcdxp,cfghyq,xyuv->abndup,nfghvq'
        scale_u_subscripts = 'absdup,s->absdup'
        scale_v_subscripts = 'sbcdvp,s->sbcdvp'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        prod_subscripts = 'abcdxp,efahyq,xyuv->nbcdup,efnhvq'
        scale_u_subscripts = 'sbcdup,s->sbcdup'
        scale_v_subscripts = 'absdvp,s->absdvp'
    elif x_pos[1] < y_pos[1]: # [x y]
        prod_subscripts = 'abcdxp,efgbyq,xyuv->ancdup,efgnvq'
        scale_u_subscripts = 'ascdup,s->ascdup'
        scale_v_subscripts = 'abcsvp,s->abcsvp'
    elif x_pos[1] > y_pos[1]: # [y x]
        prod_subscripts = 'abcdxp,edghyq,xyuv->abcnup,enghvq'
        scale_u_subscripts = 'abcsup,s->abcsup'
        scale_v_subscripts = 'ascdvp,s->ascdvp'
    else:
        assert False

    u, s, v = state.backend.einsumsvd(prod_subscripts, x, y, operator, option=svd_option)
    s = s ** 0.5
    u = state.backend.einsum(scale_u_subscripts, u, s)
    v = state.backend.einsum(scale_v_subscripts, v, s)
    state.grid[x_pos] = u
    state.grid[y_pos] = v


def apply_local_pair_operator_qr(state, operator, positions, svd_option):
    assert len(positions) == 2
    if svd_option is None:
        svd_option = ReducedSVD()
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        split_x_subscripts = 'abcdxp->abdi,icxp'
        split_y_subscripts = 'cfghyq->fghj,jcyq'
        recover_x_subscripts = 'abdi,isup,s->absdup'
        recover_y_subscripts = 'fghj,jsvq,s->sfghvq'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        split_x_subscripts = 'abcdxp->bcdi,iaxp'
        split_y_subscripts = 'efahyq->efhj,jayq'
        recover_x_subscripts = 'bcdi,isup,s->sbcdup'
        recover_y_subscripts = 'efhj,jsvq,s->efshvq'
    elif x_pos[1] < y_pos[1]: # [x y]
        split_x_subscripts = 'abcdxp->acdi,ibxp'
        split_y_subscripts = 'efgbyq->efgj,jbyq'
        recover_x_subscripts = 'acdi,isup,s->ascdup'
        recover_y_subscripts = 'efgj,jsvq,s->efgsvq'
    elif x_pos[1] > y_pos[1]: # [y x]
        split_x_subscripts = 'abcdxp->abci,idxp'
        split_y_subscripts = 'edghyq->eghj,jdyq'
        recover_x_subscripts = 'abci,isup,s->abcsup'
        recover_y_subscripts = 'eghj,jsvq,s->esghvq'
    else:
        assert False

    xq, xr = state.backend.einqr(split_x_subscripts, x)
    yq, yr = state.backend.einqr(split_y_subscripts, y)

    u, s, v = state.backend.einsumsvd('ikxp,jkyq,xyuv->isup,jsvq', xr, yr, operator, option=svd_option)
    s = s ** 0.5
    state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
    state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)


def apply_local_pair_operator_gram_qr_local(state, operator, positions, rank):
    assert len(positions) == 2
    x_pos, y_pos = positions
    x, y = state.grid[x_pos], state.grid[y_pos]

    if x_pos[0] < y_pos[0]: # [x y]^T
        gram_x_subscripts = 'abcdxp,abCdXp->xcXC'
        gram_y_subscripts = 'cfghyq,CfghYq->ycYC'
        xq_subscripts = 'abcdxp,xci->abdpi'
        yq_subscripts = 'cfghyq,ycj->fghqj'
        recover_x_subscripts = 'abdpi,isu,s->absdup'
        recover_y_subscripts = 'fghqj,jsv,s->sfghvq'
    elif x_pos[0] > y_pos[0]: # [y x]^T
        gram_x_subscripts = 'abcdxp,AbcdXp->xaXA'
        gram_y_subscripts = 'efahyq,efAhYq->yaYA'
        xq_subscripts = 'abcdxp,xai->bcdpi'
        yq_subscripts = 'efahyq,yaj->efhqj'
        recover_x_subscripts = 'bcdpi,isu,s->sbcdup'
        recover_y_subscripts = 'efhqj,jsv,s->efshvq'
    elif x_pos[1] < y_pos[1]: # [x y]
        gram_x_subscripts = 'abcdxp,aBcdXp->xbXB'
        gram_y_subscripts = 'efgbyq,efgBYq->ybYB'
        xq_subscripts = 'abcdxp,xbi->acdpi'
        yq_subscripts = 'efgbyq,ybj->efgqj'
        recover_x_subscripts = 'acdpi,isu,s->ascdup'
        recover_y_subscripts = 'efgqj,jsv,s->efgsvq'
    elif x_pos[1] > y_pos[1]: # [y x]
        gram_x_subscripts = 'abcdxp,abcDXp->xdXD'
        gram_y_subscripts = 'edghyq,eDghYq->ydYD'
        xq_subscripts = 'abcdxp,xci->abcpi'
        yq_subscripts = 'cfghyq,ycj->eghqj'
        recover_x_subscripts = 'abcpi,isu,s->abcsup'
        recover_y_subscripts = 'eghqj,jsv,s->esghvq'
    else:
        assert False

    def gram_qr_local(backend, a, gram_a_subscripts, q_subscripts):
        gram_a = backend.einsum(gram_a_subscripts, a.conj(), a)
        d, xi = gram_a.shape[:2]

        # local
        with Timer(state.backend, 'KOALA_gram_qr_local'):
            with Timer(state.backend, 'KOALA_ctf_to_numpy'):
                gram_a = gram_a.numpy().reshape(d*xi, d*xi)
            with Timer(state.backend, 'KOALA_eigh_etc'):
                w, v = la.eigh(gram_a, overwrite_a=True)
                s = np.clip(w, 0, None) ** 0.5
                s_pinv = np.divide(1, s, out=np.zeros_like(s), where=s!=0)
                r = np.einsum('j,ij->ji', s, v.conj()).reshape(d*xi, d, xi)
                r_inv = np.einsum('j,ij->ij', s_pinv, v).reshape(d, xi, d*xi)

        with Timer(state.backend, 'KOALA_numpy_to_ctf'):
            r = backend.astensor(r)
            r_inv = backend.astensor(r_inv)
        with Timer(state.backend, 'KOALA_multiply_r_inv'):
            q = backend.einsum(q_subscripts, a, r_inv)
        return q, r

    with Timer(state.backend, 'KOALA_gram_qr'):
        xq, xr = gram_qr_local(state.backend, x, gram_x_subscripts, xq_subscripts)
        yq, yr = gram_qr_local(state.backend, y, gram_y_subscripts, yq_subscripts)
    with Timer(state.backend, 'KOALA_einsumsvd'):
        u, s, v = state.backend.einsumsvd('ixk,jyk,xyuv->isu,jsv', xr, yr, operator, option=ReducedSVD(rank))
    with Timer(state.backend, 'KOALA_recover_site'):
        s = s ** 0.5
        state.grid[x_pos] = state.backend.einsum(recover_x_subscripts, xq, u, s)
        state.grid[y_pos] = state.backend.einsum(recover_y_subscripts, yq, v, s)
