"""Wrapped numpy functions.

This library contains all functions and methods in numpy. Unless specified in
'nograd_functions', the function is assumed to have a registered vector-Jacobian
product.

Uses Python namespace-is-a-dict magic.
"""
from __future__ import absolute_import
import types
from autograd.tracer import primitive, notrace_primitive
import numpy as _np

# ----- Non-differentiable functions -----

nograd_functions = [
    _np.all,
    _np.allclose,
    _np.any,
    _np.argmax,
    _np.argmin,
    _np.argpartition,
    _np.argsort,
    _np.argwhere,
    _np.around,
    _np.array_equal,
    _np.array_equiv,
    _np.ceil,
    _np.count_nonzero,
    _np.equal,
    _np.fix,
    _np.flatnonzero,
    _np.floor,
    _np.floor_divide,
    _np.greater,
    _np.greater_equal,
    _np.isclose,
    _np.iscomplex,
    _np.iscomplexobj,
    _np.isfinite,
    _np.isinf,
    _np.isnan,
    _np.isneginf,
    _np.isposinf,
    _np.isreal,
    _np.isscalar,
    _np.less,
    _np.less_equal,
    _np.logical_and,
    _np.logical_not,
    _np.logical_or,
    _np.logical_xor,
    _np.ndim,
    _np.nonzero,
    _np.not_equal,
    _np.ones_like,
    _np.result_type,
    _np.rint,
    _np.round,
    _np.searchsorted,
    _np.shape,
    _np.sign,
    _np.size,
    _np.trunc,
    _np.zeros_like,
]

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

def wrap_namespace(old, new):
    """Copy all functions in 'old' namespace to 'new' namespace.

    Args:
      old: __dict__ of module to copy from.
      new: __dict__ of module to copy to.
    """
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if obj in nograd_functions:
            # Functions without gradients. We don't bother to trace values that
            # enter here.
            new[name] = notrace_primitive(obj)
        elif type(obj) in function_types:
            # Functions with gradients. We trace values.
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            # Wrap int types with something identical except that calls to __new__
            # immediately strip argument of boxes.
            #
            # TODO(duckworthd): Why do numpy int types need to be boxed, but not
            # Python's int()?
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

# Set autograd.numpy.<function> = wrap(numpy.<function>)
wrap_namespace(_np.__dict__, globals())
