from collections import defaultdict
from contextlib import contextmanager

from .util import subvals, wraps

def trace(start_node, fun, x):
    with trace_stack.new_trace() as trace_id:
        start_box = new_box(x, trace_id, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace_id == start_box._trace_id:
            return end_box._value, end_box._node
        else:
            # Output seems independent of input
            return end_box, None

class Node(object):
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        self.recipe = (fun, value, args, kwargs, parent_argnums)

    def initialize_root(self):
        self.parents = []
        self.recipe = (lambda x: x, None, (), {}, [])

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root

def primitive(f_raw):
    """Wraps a function so that its gradient (vjp) can be specified and its
    invocation can be recorded."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        boxed_args, trace_id = find_top_boxed_args(args)
        if boxed_args:
            argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
            parents = tuple(box._node for _, box in boxed_args)
            argnums = tuple(argnum for argnum, _ in boxed_args)
            ans = f_wrapped(*argvals, **kwargs)
            node = Node(ans, f_wrapped, argvals, kwargs, argnums, parents)
            return new_box(ans, trace_id, node)
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped

def notrace_primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    return f_wrapped

def find_top_boxed_args(args):
    top_trace_id = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if isbox(arg):
            if arg._trace_id > top_trace_id:
                top_boxes = [(argnum, arg)]
                top_trace_id = arg._trace_id
            elif arg._trace_id == top_trace_id:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace_id

class TraceStack(object):
    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1
trace_stack = TraceStack()

class Box(object):
    type_mappings = {}
    types = set()

    def __init__(self, value, trace_id, node):
        self._value = value
        self._node = node
        self._trace_id = trace_id

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

    @classmethod
    def register(cls, value_type):
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls
        Box.type_mappings[cls] = cls


box_type_mappings = Box.type_mappings
def new_box(value, trace, node):
    try:
        return box_type_mappings[type(value)](value, trace, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types
isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
