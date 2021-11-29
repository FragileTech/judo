from typing import Callable

import numpy

from judo.data_types import dtype
from judo.judo_backend import Backend, torch
from judo.typing import Tensor


def to_backend_wrap(func) -> Callable[..., Tensor]:
    def wrapped(*args, **kwargs):
        return to_backend(func(*args, **kwargs))

    return wrapped


def is_floating_tensor(x):
    if isinstance(x, numpy.ndarray):
        return isinstance(x.flat[0], (numpy.floating, float))
    elif isinstance(x, torch.Tensor):
        return x.is_floating_point()
    return False


def is_floating_object(x):
    return (
        is_floating_tensor(x)
        or isinstance(x, float)
        or (isinstance(x, (tuple, list)) and isinstance(x[0], float))
    )


def _new_torch_tensor_copy(x, dtype=None, device=None, requires_grad=None, pin_memory=False):
    # TODO(guillemdb): benchmark different alternatives and implement the fastest
    requires_grad = requires_grad and is_floating_object(x)
    return torch.tensor(
        x,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
    )


def _new_torch_tensor_avoid_copy(x, dtype=None, device=None, requires_grad=None):
    if isinstance(x, torch.Tensor):
        y = x.to(dtype=dtype, device=device)
    # elif isinstance(x, numpy.ndarray):
    #    y = torch.from_numpy(x)
    #    y = (y if (device is None and dtype is None) else
    #    y.to(device=device, dtype=dtype, copy=False))
    elif isinstance(x, torch.autograd.grad_mode.no_grad):
        return x
    else:
        y = torch.as_tensor(x, dtype=dtype, device=device)
    return (
        y
        if requires_grad is not None or not requires_grad or not is_floating_tensor(y)
        else y.requires_grad_()
    )


def new_torch_tensor(x, dtype=None, device=None, requires_grad=None, copy=False, pin_memory=False):
    if copy:
        return _new_torch_tensor_copy(
            x,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    return _new_torch_tensor_avoid_copy(x, dtype=dtype, device=device, requires_grad=requires_grad)


def new_numpy_array(x, dtype=None, copy=False, allow_objects: bool = False):
    try:
        return numpy.array(x, dtype=dtype, copy=copy)
    except Exception as e:
        if allow_objects:
            return numpy.array(tuple(x), dtype=object, copy=copy)
        raise e


def new_backend_tensor(
    x,
    dtype=None,
    device=None,
    requires_grad: bool = None,
    copy: bool = False,
    pin_memory: bool = False,
):
    kwargs = update_with_backend_values(
        requires_grad=requires_grad,
        device=device,
        copy=copy,
        dtype=dtype,
    )
    if Backend.is_numpy():
        return new_numpy_array(x, dtype=kwargs.get("dtype"), copy=kwargs.get("copy"))
    elif Backend.is_torch():
        return new_torch_tensor(
            x,
            dtype=kwargs.get("dtype"),
            copy=kwargs.get("copy"),
            device=kwargs.get("device"),
            requires_grad=kwargs.get("requires_grad"),
            pin_memory=pin_memory,
        )


def copy_torch(x: torch.Tensor, requires_grad, device=None) -> torch.Tensor:
    grad = requires_grad if requires_grad is not None else Backend.requires_grad()
    device = torch.device(device) if device is not None else Backend.get_device()
    new_tensor = x.clone()
    if device is not None and new_tensor.device == device:
        new_tensor = new_tensor.to(device)
    if grad is not None and not grad:
        new_tensor = new_tensor.detach()
    return new_tensor


def astype(x, dtype):
    funcs = {
        "numpy": lambda x: x.astype(dtype),
        "torch": lambda x: x.to(dtype),
    }
    return Backend.execute(x, funcs)


def as_tensor(x, *args, **kwargs):
    funcs = {
        "numpy": lambda x: numpy.ascontiguousarray(x, *args, **kwargs),
        "torch": lambda x: torch.as_tensor(x, *args, **kwargs),
    }
    return Backend.execute(x, funcs)


def to_numpy(x, copy=False):
    if isinstance(x, numpy.ndarray):
        return x.copy() if copy else x
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return new_numpy_array(x, copy=copy)


def _assign_device_and_grad(x, device, requires_grad):
    if device is not None:
        x = x.to(device=device)
    if requires_grad is not None and x.requires_grad != requires_grad and is_floating_tensor(x):
        x = x.requires_grad_(requires_grad)
    return x


def to_torch(x, requires_grad: bool = None, device: str = None, copy: bool = False):
    use_grad = requires_grad if requires_grad is not None else Backend.requires_grad()
    device = device if device is not None else Backend.get_device()
    if isinstance(x, torch.Tensor):
        return (
            copy_torch(x, device=device, requires_grad=use_grad)
            if copy
            else _assign_device_and_grad(x, device=device, requires_grad=use_grad)
        )
    return new_torch_tensor(x, device=device, copy=copy)


def update_with_backend_values(**kwargs):
    user_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    backend_kwargs = Backend.get_backend_state()
    backend_kwargs.update(user_kwargs)
    return backend_kwargs


def to_backend(
    x: "Tensor",
    requires_grad: bool = None,
    device: str = None,
    copy: bool = None,
) -> Tensor:
    kwargs = update_with_backend_values(requires_grad=requires_grad, device=device, copy=copy)
    if Backend.is_numpy():
        return to_numpy(x, kwargs.get("copy"))
    return to_torch(
        x,
        requires_grad=kwargs.get("requires_grad"),
        device=kwargs.get("device"),
        copy=kwargs.get("copy"),
    )


def match_backend(x, other) -> Tensor:
    if isinstance(x, numpy.ndarray):
        return to_numpy(other)
    elif isinstance(x, torch.Tensor):
        return torch.tensor(other)


class JudoTensor:
    def __new__(
        cls,
        x,
        dtype=None,
        device: str = None,
        requires_grad: bool = None,
        copy: bool = None,
        pin_memory: bool = False,
    ) -> Tensor:
        return new_backend_tensor(
            x,
            requires_grad=requires_grad,
            device=device,
            copy=copy,
            pin_memory=pin_memory,
            dtype=dtype,
        )


tensor = JudoTensor
array = JudoTensor


def copy(x, requires_grad: bool = None):
    if x is None:
        return
    if not dtype.is_tensor(x):
        x = JudoTensor(x)

    funcs = {
        "numpy": lambda x: x.copy(),
        "torch": lambda x: copy_torch(x, requires_grad),
    }
    return Backend.execute(x, funcs)
