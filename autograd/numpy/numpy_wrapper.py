from __future__ import absolute_import
import types
from autograd.tracer import primitive, notrace_primitive
import numpy as _np

# ----- Non-differentiable functions -----

nograd_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type, _np.zeros_like,
    _np.ones_like, _np.floor, _np.ceil, _np.round, _np.rint, _np.around,
    _np.fix, _np.trunc, _np.all, _np.any, _np.argmax, _np.argmin,
    _np.argpartition, _np.argsort, _np.argwhere, _np.nonzero, _np.flatnonzero,
    _np.count_nonzero, _np.searchsorted, _np.sign, _np.ndim, _np.shape,
    _np.floor_divide, _np.logical_and, _np.logical_or, _np.logical_not,
    _np.logical_xor, _np.isfinite, _np.isinf, _np.isnan, _np.isneginf,
    _np.isposinf, _np.allclose, _np.isclose, _np.array_equal, _np.array_equiv,
    _np.greater, _np.greater_equal, _np.less, _np.less_equal, _np.equal,
    _np.not_equal, _np.iscomplexobj, _np.iscomplex, _np.size, _np.isscalar,
    _np.isreal, _np.zeros_like, _np.ones_like, _np.result_type
]

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if obj in nograd_functions:
            new[name] = notrace_primitive(obj)
        elif type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_np.__dict__, globals())
