import numpy
import pytest


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from judo import Backend, dtype


@pytest.fixture()
def backend():
    return Backend


backend_names = ["numpy", "torch"] if TORCH_AVAILABLE else ["numpy"]


class TestDataTypes:
    @pytest.mark.parametrize("name", backend_names)
    def test_bool(self, backend, name):
        backend.set_backend(name)
        if name == "numpy":
            assert dtype.bool == numpy.bool_, dtype.bool
        elif name == "torch":
            assert dtype.bool == torch.bool
