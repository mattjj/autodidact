"""Utility functions."""


def subvals(x, ivs):
    """Replace the i-th value of x with v.

    Args:
      x: iterable of items.
      ivs: list of (int, value) pairs.

    Returns:
      x modified appropriately.
    """
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def subval(x, i, v):
    """Replace the i-th value of x with v."""
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):
    """Decorator for a function wrapping another.

    Used when wrapping a function to ensure its name and docstring get copied
    over.

    Args:
      fun: function to be wrapped
      namestr: Name string to use for wrapped function.
      docstr: Docstring to use for wrapped function.
      **kwargs: additional string format values.

    Return:
      Wrapped function.
    """
    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun), **kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun), doc=get_doc(fun), **kwargs)
        finally:
            return f
    return _wraps

def wrap_nary_f(fun, op, argnum):
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """
    return wraps(fun, namestr, docstr, op=get_name(op), argnum=argnum)

get_name = lambda f: getattr(f, '__name__', '[unknown name]')
get_doc  = lambda f: getattr(f, '__doc__' , '')
