"""Convenience functions built on top of `make_vjp`."""

import numpy as np

from .util import unary_to_nary
from .core import make_vjp as _make_vjp

make_vjp = unary_to_nary(_make_vjp)

@unary_to_nary
def grad(fun, x):
    vjp, ans = _make_vjp(fun, x)
    return vjp(np.ones_like(ans))
