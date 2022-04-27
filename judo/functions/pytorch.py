from judo.data_types import dtype
from judo.judo_backend import torch
from judo.judo_tensor import astype, tensor


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
    "ones",
    "zeros",
    "arange",
    "full_like",
    "allclose",
]


def parse_dim(axis, dim):
    if axis is not None:
        return axis
    return dim


def concatenate(x, axis=0, out=None, dim=0):
    axis = parse_dim(axis, dim)
    return torch.cat(x, dim=axis, out=out)


def stack(x, axis=0, out=None, dim=0):
    axis = parse_dim(axis, dim)
    return torch.stack(x, dim=axis, out=out)


def clip(x, a_min, a_max, out=None):

    _tensor = min(x, other=astype(tensor(a_max), dtype=x.dtype))
    return max(_tensor, other=astype(tensor(a_min), dtype=x.dtype), out=out)


def repeat(x, repeat, axis=None, dim=None):
    axis = parse_dim(axis, dim)
    return torch.repeat_interleave(x, repeat, dim=axis)


def tile(x, repeat):
    return torch.Tensor.repeat(x, *repeat if isinstance(repeat, tuple) else [repeat])


def norm(x, ord=None, axis=None, keepdims=False, dim=None):
    axis = parse_dim(axis, dim)
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def min(x, axis=None, other=None, out=None, dim=None):
    axis = parse_dim(axis, dim)
    if other is None:
        axis = axis if axis is not None else 0
        val, _ = torch.min(x, dim=axis, out=out)
        return val
    return torch.min(x, other=other, out=out)


def max(x, axis=None, other=None, out=None, dim=None):
    axis = parse_dim(axis, dim)
    if other is None:
        axis = axis if axis is not None else 0
        val, _ = torch.max(x, dim=axis, out=out)
        return val
    return torch.max(x, other=other, out=out)


def unsqueeze(x, axis=0, dim=0):
    axis = parse_dim(axis, dim)
    return x.unsqueeze(axis)


def where(cond, a, b, *args, **kwargs):
    was_bool = False
    if a.dtype == dtype.bool:
        a = a.to(dtype.int32)
        was_bool = True
    if b.dtype == dtype.bool:
        b = b.to(dtype.int32)
        was_bool = True
    res = torch.where(cond, a, b)
    return res.to(dtype.bool) if was_bool else res


def argmax(x, axis=None, dim=None, *args, **kwargs):
    axis = parse_dim(axis, dim)
    # where not implemented for bool in
    axis = axis if axis is not None else 0
    was_bool = False
    if x.dtype == dtype.bool:
        x = x.to(dtype.int32)
        was_bool = True
    res = torch.argmax(x, dim=axis, *args, **kwargs)
    return res.to(dtype.bool) if was_bool else res


def sqrt(x, *args, **kwargs):
    # where not implemented for bool in
    was_bool = False

    if x.dtype in (dtype.int64, dtype.int32):
        _x = x.to(dtype.float32)
        was_bool = True
    else:
        _x = x
    res = torch.sqrt(_x, *args, **kwargs)
    return res.to(x.dtype) if was_bool else res


def logical_or(a, b):
    return a + b


def logical_and(a, b):
    return a & b


def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


def arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def abs(*args, **kwargs):
    return torch.abs(*args, **kwargs)


def allclose(*args, **kwargs):
    return torch.allclose(*args, **kwargs)


def full_like(*args, **kwargs):
    return torch.full_like(*args, **kwargs)


def mod(*args, **kwargs):
    return torch.fmod(*args, **kwargs)
