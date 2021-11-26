import pytest

from judo import Backend


@pytest.fixture()
def backend():
    return Backend


class TestBackend:
    def test_init(self, backend):
        pass

    def test_set_backend(self, backend):
        assert backend.is_numpy()
        backend.set_backend("torch")
        assert backend.is_torch()

        assert backend.get_device() == "cpu"
        backend.set_backend(device="cuda")
        assert backend.get_device() == "cuda"

        assert not backend.requires_grad()
        backend.set_backend(requires_grad=True)
        assert backend.requires_grad()

        backend.reset_state()
        assert backend.is_numpy()
        assert backend.get_device() == "cpu"
        assert not backend.requires_grad()
