import tensorbackends

from .candecomp import CanonicalDecomp
from .utils import initialize_random_factors


def random(nsite, rank, backend='numpy'):
    backend = tensorbackends.get(backend)
    factors = initialize_random_factors(rank, nsite, backend)
    return CanonicalDecomp(factors, backend)
