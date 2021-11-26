from contextlib import contextmanager
import os

import numpy
import yaml


try:
    import torch
except ImportError:

    class cuda:
        @staticmethod
        def is_available():
            return False

    class torch_random:
        @staticmethod
        def manual_seed(*args, **kwargs):
            return None

    class torch:
        cuda = cuda
        Tensor = numpy.ndarray
        random = torch_random


FALLBACK_DEFAULTS = {
    "backend": "numpy",
    "device": "cpu",
    "requires_grad": None,
    "true_hash": False,
    "copy": False,
}
config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")


def load_backend_config(filepath=config_file):
    try:
        with open(filepath, "r") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        with open(filepath, "w") as file:
            yaml.dump(FALLBACK_DEFAULTS, file)
        config = FALLBACK_DEFAULTS
    backend = config.get("backend", FALLBACK_DEFAULTS["backend"])
    device = config.get("device", FALLBACK_DEFAULTS["device"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif str(device).lower() == "none":
        device = None
    requires_grad = config.get("requires_grad", FALLBACK_DEFAULTS["requires_grad"])
    true_hash = config.get("true_hash", FALLBACK_DEFAULTS["true_hash"])
    copy = config.get("copy", FALLBACK_DEFAULTS["copy"])
    return backend, device, requires_grad, true_hash, copy


def update_backend_config(
    backend: str = None,
    device: str = None,
    requires_grad: bool = None,
    filepath=config_file,
    true_hash: bool = None,
    copy: bool = None,
):
    with open(filepath, "r") as stream:
        config = yaml.safe_load(stream)
        if config is None:
            config = {}
    if backend is not None:
        config["default_backend"] = backend
    if device is not None:
        config["default_device"] = device
    if requires_grad is not None:
        config["requires_grad"] = requires_grad
    if true_hash is not None:
        config["true_hash"] = true_hash
    if copy is not None:
        config["copy"] = copy
    with open(filepath, "w") as outfile:
        yaml.dump(config, outfile)


@contextmanager
def _use_backend(cls, name, device=None, requires_grad=None, copy=None):
    if name is not None:
        cls._check_valid_backend(name)
    curr_state = cls.get_backend_state()
    cls.set_backend(name=name, device=device, requires_grad=requires_grad, copy=copy)
    try:
        yield
    finally:
        cls.set_backend(**curr_state)


@contextmanager
def use_backend(name, device=None, requires_grad=None):
    if name is not None:
        Backend._check_valid_backend(name)
    curr_state = Backend.get_backend_state()
    Backend.set_backend(name=name, device=device, requires_grad=requires_grad)
    try:
        yield
    finally:
        Backend.set_backend(**curr_state)


_backend, _device, _requires_grad, _true_hash, _copy = load_backend_config()
_backend, _device, _requires_grad, _true_hash, _copy = (
    str(_backend),
    str(_device) if _device is not None else None,
    bool(_requires_grad) if _requires_grad is not None else None,
    bool(_true_hash),
    bool(_copy),
)


class Backend:
    AVAILABLE_BACKENDS = ["numpy", "torch"]
    _backend, _device, _requires_grad, _true_hash = _backend, _device, _requires_grad, _true_hash
    _copy = _copy

    @classmethod
    def _check_valid_backend(cls, name):
        if name not in cls.AVAILABLE_BACKENDS:
            raise ValueError(
                "%s not supported. Available backends: %s" % (name, cls.AVAILABLE_BACKENDS),
            )

    @classmethod
    def get_backend_state(cls):
        state = {
            "name": str(cls._backend),
            "device": str(cls._device) if cls._device is not None else None,
            "requires_grad": bool(cls._requires_grad) if cls._requires_grad is not None else None,
            "true_hash": bool(cls._true_hash),
            "copy": bool(cls._copy),
        }
        return state

    @classmethod
    def get_current_backend(cls):
        return cls._backend

    @classmethod
    def get_device(cls):
        if cls._device is not None:
            return str(cls._device)
        return None

    @classmethod
    def requires_grad(cls):
        return cls._requires_grad

    @classmethod
    def use_true_hash(cls):
        return cls._true_hash

    @classmethod
    def set_defaults(
        cls,
        name=None,
        device=None,
        requires_grad: bool = None,
        copy: bool = None,
        true_hash: bool = None,
        set_backend: bool = True,
    ):
        update_backend_config(
            backend=name,
            device=device,
            requires_grad=requires_grad,
            true_hash=true_hash,
            copy=copy,
        )
        if set_backend:
            cls.set_backend(
                name=name,
                device=device,
                requires_grad=requires_grad,
                copy=copy,
                true_hash=true_hash,
            )

    @classmethod
    def set_backend(
        cls,
        name=None,
        device=None,
        requires_grad: bool = None,
        copy: bool = None,
        true_hash: bool = None,
    ):
        if name is not None:
            cls._check_valid_backend(name)
            cls._backend = name
        if device is not None:
            cls._device = device
        if requires_grad is not None:
            cls._requires_grad = requires_grad
        if copy is not None:
            cls._copy = copy
        if true_hash is not None:
            cls._true_hash = true_hash

    @classmethod
    def is_numpy(cls):
        return cls._backend == "numpy"

    @classmethod
    def is_torch(cls):
        return cls._backend == "torch"

    @classmethod
    def can_use_cuda(cls):
        return cls.is_torch() and cls._device != "cpu"

    @classmethod
    def execute(cls, value, funcs):
        backend = cls.get_current_backend()
        return funcs[backend](value)

    @classmethod
    def use_backend(cls, name=None, device=None, requires_grad=None, copy=None):
        return _use_backend(cls, name=name, device=device, requires_grad=requires_grad, copy=copy)

    @classmethod
    def reset_state(cls):
        global _backend, _device, _requires_grad, _true_hash, _copy
        _backend, _device, _requires_grad, _true_hash, _copy = load_backend_config()
        _backend, _device, _requires_grad, _true_hash, _copy = (
            str(_backend),
            str(_device) if _device is not None else None,
            bool(_requires_grad) if _requires_grad is not None else None,
            bool(_true_hash),
            bool(_copy),
        )
        cls._backend, cls._device, cls._requires_grad, cls._true_hash, cls._copy = (
            _backend,
            _device,
            _requires_grad,
            _true_hash,
            _copy,
        )
