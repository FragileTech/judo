import pytest

from judo import dtype
from judo.data_structures.states import States
from judo.functions.api import API


states_classes = [States]


@pytest.fixture(scope="class", params=states_classes)
def states_class(request):
    return request.param


class TestStates:
    def test_init_dict(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": dtype.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        assert new_states.n == 2

    def test_init_kwargs(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states._batch_size == 2
        assert name in new_states.keys()
        assert getattr(new_states, name) == name, type(new_states)

    def test_getitem(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states[name] == name, type(new_states)

    def test_setitem(self, states_class):
        name_1 = "miau"
        val_1 = name_1
        name_2 = "elephant"
        val_2 = API.arange(10)
        new_states = states_class(batch_size=2)
        new_states[name_1] = val_1
        new_states[name_2] = val_2
        assert new_states[name_1] == val_1, type(new_states)
        assert (new_states[name_2] == val_2).all(), type(new_states)

    def test_repr(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert isinstance(new_states.__repr__(), str)

    def test_n(self, states_class):
        new_states = states_class(batch_size=2)
        assert new_states.n == new_states._batch_size == 2

    def test_get(self, states_class):
        new_states = states_class(batch_size=2, test="test")
        assert new_states.get("test") == "test"
        assert new_states.get("AKSJDFKG") is None
        assert new_states.get("ASSKADFKA", 5) == 5

    def test_split_states(self, states_class):
        batch_size = 20
        new_states = states_class(batch_size=batch_size, test="test")
        for s in new_states.split_states(batch_size):
            assert len(s) == 1
            assert s.test == "test"
        data = API.repeat(API.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        for s in new_states.split_states(batch_size):
            assert len(s) == 1
            assert s.test == "test"
            assert bool((s.data == API.arange(5)).all()), s.data
        chunk_len = 4
        test_data = API.repeat(API.arange(5).reshape(1, -1), chunk_len, 0)
        for s in new_states.split_states(5):
            assert len(s) == chunk_len
            assert s.test == "test"
            assert (s.data == test_data).all(), (s.data.shape, test_data.shape)

        batch_size = 21
        data = API.repeat(API.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        chunk_len = 5
        test_data = API.repeat(API.arange(5).reshape(1, -1), chunk_len, 0)
        split_states = list(new_states.split_states(5))
        for s in split_states[:-1]:
            assert len(s) == chunk_len
            assert s.test == "test"
            assert (s.data == test_data).all(), (s.data.shape, test_data.shape)

        assert len(split_states[-1]) == 1
        assert split_states[-1].test == "test"
        assert (split_states[-1].data == API.arange(5)).all(), (s.data.shape, test_data.shape)

    def test_get_params_dir(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": dtype.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        params_dict = new_states.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki, _ in v.items():
                assert isinstance(ki, str)

    def test_merge_states(self, states_class):
        batch_size = 21
        data = API.repeat(API.arange(5).reshape(1, -1), batch_size, 0)
        new_states = states_class(batch_size=batch_size, test="test", data=data)
        split_states = tuple(new_states.split_states(batch_size))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()

        split_states = tuple(new_states.split_states(5))
        merged = new_states.merge_states(split_states)
        assert len(merged) == batch_size
        assert merged.test == "test"
        assert (merged.data == data).all()
