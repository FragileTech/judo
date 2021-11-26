import pytest

import judo
from judo import Backend


@pytest.fixture()
def backend():
    return Backend


class TestAPI:
    def test_from_judo(self, backend):
        x = judo.zeros((10, 10))
        assert judo.sqrt(x).sum() == 0
