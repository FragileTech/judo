import numpy
import pytest
import torch

from judo import Backend, dtype


@pytest.fixture()
def backend():
    return Backend


class TestDataTypes:
    def test_bool(self, backend):
        backend.set_backend("numpy")
        assert dtype.bool == numpy.bool_, dtype.bool
        backend.set_backend("torch")
        assert dtype.bool == torch.bool
