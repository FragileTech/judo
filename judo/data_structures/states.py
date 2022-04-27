import copy
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy

import judo
from judo.functions.api import API
from judo.functions.hashing import hasher
from judo.judo_tensor import tensor
from judo.typing import StateDict, Tensor


def params_to_tensor(
    n_walkers: int,
    shape: Optional[tuple] = None,
    size: Optional[tuple] = None,
    **params: Dict[str, Any],
) -> Tensor:
    """
    Create a dictionary containing the arrays specified by param_dict.

    Args:
        shape: Shape
        params: Dictionary defining the additional parameters for initializing the tensor.
        n_walkers: Number items in the first dimension of the data tensors.
        size: Shape of the target tensor without accounting for the batch_size dimension.

    Returns:
          Dictionary with the same keys as param_dict, containing arrays specified \
          by `param_dict` values.

    """

    # Shape already includes the number of walkers. Remove walkers axis to create size.
    if shape is not None and len(shape) > 1:
        val_size = shape[1:]
    else:
        val_size = size
    # Create appropriate shapes with current state's number of walkers.
    sizes = n_walkers if val_size is None else tuple([n_walkers]) + val_size
    param_tensor = API.zeros(sizes, **params)
    return param_tensor


class States:
    """
    Data structure that handles teh data that defines a population of agents.

    Each population attribute will be stored as a tensor with its first dimension \
    (batch size) representing each agent.

    In order to define a tensor attribute, a `state_dict` dictionary needs to be \
    specified using the following structure::

        state_dict = {"name_1": {"size": tuple([1]),
                                 "dtype": numpy.float32,
                                 "is_array": True,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor. The \
    created arrays will accessible the ``name_1`` attribute of the class, or \
    indexing the class with ``states["name_1"]``.

    If ``size`` is not defined the attribute will be considered a vector of \
    length `batch_size`.


    Args:
        batch_size: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: Data can be directly specified as keyword arguments.

    """

    def __init__(self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs):
        """
        Initialize a :class:`States`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        attr_dict = self.params_to_arrays(state_dict, batch_size) if state_dict is not None else {}
        attr_dict.update(kwargs)
        self._tensor_names = [k for k, v in attr_dict.items() if judo.is_tensor(v)]
        self._names = tuple(attr_dict.keys())
        self._attr_dict = attr_dict
        self.update(**self._attr_dict)
        self._batch_size = batch_size

    def __len__(self) -> int:
        """Length is equal to batch_size."""
        return self._batch_size

    def __getitem__(self, item: Union[str, int]) -> Union[Tensor, List[Tensor], "States"]:
        """
        Query an attribute of the class as if it was a dictionary.

        Args:
            item: Name of the attribute to be selected.

        Returns:
            The corresponding item.

        """
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except AttributeError:
                raise TypeError("Tried to get a non existing attribute with key {}".format(item))
        elif judo.is_int(item):
            return self._ix(item)
        else:
            raise TypeError(
                "item must be an instance of str, got {} of type {} instead".format(
                    item,
                    type(item),
                ),
            )

    def _ix(self, index: int):
        # TODO(guillemdb): Allow slicing
        data = {k: API.unsqueeze(v[index], 0) if judo.is_tensor(v) else v for k, v in self.items()}
        return self.__class__(batch_size=1, **data)

    def __setitem__(self, key, value: Union[Tuple, List, Tensor]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None.

        """
        if key not in self._names:
            self._names += (key,)
        self.update(**{key: value})

    def __repr__(self):
        string = "{} with {} walkers\n".format(self.__class__.__name__, self.n)
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, type(v), shape)
            string += new_str
        return string

    def __hash__(self) -> int:
        return hasher.hash_state(self)

    def group_hash(self, name: str) -> int:
        """Return a unique id for a given attribute."""
        return hasher.hash_tensor(self[name])

    def hash_attribute(self, name: str) -> List[int]:
        """Return a unique id for each walker attribute."""
        return hasher.hash_iterable(self[name])

    @staticmethod
    def merge_states(states: Iterable["States"]) -> "States":
        """
        Combine different states containing the same kind of data into a single \
        :class:`State` with batch size equal to the sum of all the state batch \
        sizes.

        Args:
            states: Iterable returning :class:`States` with the same attributes.

        Returns:
            :class:`States` containing the combined data of the input values.

        """

        def merge_one_name(states_list, name):
            vals = []
            for state in states_list:
                data = state[name]
                # Attributes that are not tensors are not stacked.
                if not judo.is_tensor(data):
                    return data
                state_len = len(state)
                if len(data.shape) == 0 and state_len == 1:
                    # Name is scalar vector. Data is typing.Scalar value. Transform to array first
                    value = tensor([data]).flatten()
                elif len(data.shape) == 1 and state_len == 1:
                    if data.shape[0] == 1:
                        # Name is typing.Scalar vector. Data already transformed to an array
                        value = data
                    else:
                        # Name is a matrix of vectors. Data needs an additional dimension
                        value = tensor([data])
                elif len(data.shape) == 1 and state_len > 1:
                    # Name is a typing.Scalar vector. Data already has is a one dimensional array
                    value = data
                elif (
                    len(data.shape) > 1
                    and state_len > 1
                    or len(data.shape) > 1
                    and len(state) == 1
                ):
                    # Name is a matrix of vectors. Data has the correct shape
                    value = data
                else:
                    raise ValueError(
                        "Could not infer data concatenation for attribute %s  with shape %s"
                        % (name, data.shape),
                    )
                vals.append(value)
            return API.concatenate(vals)

        # Assumes all states have the same names.
        data = {name: merge_one_name(states, name) for name in states[0]._names}
        batch_size = sum(s.n for s in states)
        return states[0].__class__(batch_size=batch_size, **data)

    @property
    def n(self) -> int:
        """Return the batch_size of the vectors, which is equivalent to the number of walkers."""
        return self._batch_size

    def get(self, key: str, default=None):
        """
        Get an attribute by key and return the default value if it does not exist.

        Args:
            key: Attribute to be recovered.
            default: Value returned in case the attribute is not part of state.

        Returns:
            Target attribute if found in the instance, otherwise returns the
             default value.

        """
        if key not in self.keys():
            return default
        return self[key]

    def keys(self) -> Generator:
        """Return a generator for the attribute names of the stored data."""
        return (name for name in self._names if not name.startswith("_"))

    def vals(self) -> Generator:
        """Return a generator for the values of the stored data."""
        return (self[name] for name in self._names if not name.startswith("_"))

    def items(self) -> Generator:
        """Return a generator for the attribute names and the values of the stored data."""
        return ((name, self[name]) for name in self._names if not name.startswith("_"))

    def itervals(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n <= 1:
            yield self.vals()
            raise StopIteration
        for i in range(self.n):
            yield tuple([v[i] for v in self.vals()])

    def iteritems(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n < 1:
            return self.vals()
        for i in range(self.n):
            values = (v[i] if judo.is_tensor(v) else v for v in self.vals())
            yield tuple(self._names), tuple(values)

    def split_states(self, n_chunks: int) -> Generator["States", None, None]:
        """
        Return a generator for n_chunks different states, where each one \
        contain only the data corresponding to one walker.
        """

        def get_chunck_size(state, start, end):
            for name in state._names:
                attr = state[name]
                if judo.is_tensor(attr):
                    return len(attr[start:end])
            return int(numpy.ceil(self.n / n_chunks))

        for start, end in API.similiar_chunks_indexes(self.n, n_chunks):
            chunk_size = get_chunck_size(self, start, end)
            data = {k: val[start:end] if judo.is_tensor(val) else val for k, val in self.items()}
            new_state = self.__class__(batch_size=chunk_size, **data)
            yield new_state

    def update(self, other: "States" = None, **kwargs):
        """
        Modify the data stored in the States instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as key value attributes, \
                     where key is the name of the attribute to be updated, and value \
                      is the new value for the attribute.
        """

        def update_or_set_attributes(attrs: Union[dict, States]):
            for name, val in attrs.items():
                setattr(self, name, val)

        if other is not None:
            update_or_set_attributes(other)
        if kwargs:
            update_or_set_attributes(kwargs)

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`States`."""
        return {
            k: {"shape": v.shape, "dtype": v.dtype}
            for k, v in self.__dict__.items()
            if judo.is_tensor(v)
        }

    def copy(self) -> "States":
        """Crete a copy of the current instance."""
        param_dict = {
            str(name): judo.copy(val) if judo.is_tensor(val) else copy.deepcopy(val)
            for name, val in self.items()
        }
        return States(batch_size=self.n, **param_dict)

    @staticmethod
    def params_to_arrays(param_dict: StateDict, n_walkers: int) -> Dict[str, Tensor]:
        """
        Create a dictionary containing the arrays specified by param_dict.

        Args:
            param_dict: Dictionary defining the attributes of the tensors.
            n_walkers: Number items in the first dimension of the data tensors.

        Returns:
              Dictionary with the same keys as param_dict, containing arrays specified \
              by `param_dict` values.

        """
        tensor_dict = {}
        for key, val in param_dict.items():
            # Shape already includes the number of walkers. Remove walkers axis to create size.
            shape = val.get("shape")
            if shape is None:
                val_size = val.get("size")
            elif len(shape) > 1:
                val_size = shape[1:]
            else:
                val_size = val.get("size")
            # Create appropriate shapes with current state's number of walkers.
            sizes = n_walkers if val_size is None else tuple([n_walkers]) + val_size
            if "size" in val:
                del val["size"]
            if "shape" in val:
                del val["shape"]
            tensor_dict[key] = API.zeros(sizes, **val)
        return tensor_dict
