import numpy


AVAILABLE_FUNCTIONS = [
    "abs",
    "argmax",
    "hash_numpy",
    "hash_tensor",
    "concatenate",
    "stack",
    "clip",
    "repeat",
    "min",
    "max",
    "norm",
    "unsqueeze",
    "where",
    "sqrt",
    "tile",
    "logical_or",
    "logical_and",
    "logical_not",
    "ones",
    "zeros",
    "arange",
    "full_like",
    "allclose",
]


def concatenate(x, axis=0, out=None):
    return numpy.concatenate(x, axis=axis, out=out)


def stack(x, axis=0, out=None):
    return numpy.stack(x, axis=axis, out=out)


def clip(x, a_min, a_max, out=None):
    return numpy.clip(x, a_min, a_max, out=out)


def repeat(x, repeat, axis=None):
    return numpy.repeat(x, repeat, axis=axis)


def tile(x, repeat):
    return numpy.tile(x, repeat)


def norm(x, ord=None, axis=None, keepdims=False):
    return numpy.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def min(x, axis=None, other=None, out=None):
    if other is None:
        return numpy.min(x, axis=axis, out=out)
    else:
        return numpy.minimum(x, other, out=out)


def max(tensor, axis=None, other=None, out=None):
    if other is None:
        return numpy.max(tensor, axis=axis, out=out)
    else:
        return numpy.maximum(tensor, other, out=out)


def unsqueeze(x, axis=0):
    return numpy.expand_dims(x, axis=axis)


def where(cond, a, b, *args, **kwargs):
    return numpy.where(cond, a, b, *args, **kwargs)


def argmax(x, axis=None, *args, **kwargs):
    return numpy.argmax(x, axis=axis, *args, **kwargs)


def sqrt(x, *args, **kwargs):
    return numpy.sqrt(x, *args, **kwargs)


def logical_or(a, b):
    return numpy.logical_or(a, b)


def logical_and(a, b):
    return numpy.logical_and(a, b)


def logical_not(x):
    return numpy.logical_not(x)


def ones(*args, **kwargs):
    return numpy.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    return numpy.zeros(*args, **kwargs)


def arange(*args, **kwargs):
    return numpy.arange(*args, **kwargs)


def abs(*args, **kwargs):
    return numpy.abs(*args, **kwargs)


def allclose(*args, **kwargs):
    return numpy.allclose(*args, **kwargs)


def full_like(*args, **kwargs):
    return numpy.full_like(*args, **kwargs)


def mod(*args, **kwargs):
    return numpy.mod(*args, **kwargs)
