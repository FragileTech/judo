import pytest

from judo import API, Backend


@pytest.fixture()
def backend():
    return Backend


class TestAPI:
    def test_from_judo(self, backend):
        x = API.zeros((10, 10))
        assert API.sqrt(x).sum() == 0
