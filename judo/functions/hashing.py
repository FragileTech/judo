import uuid

import numpy
import xxhash

import judo
from judo.judo_backend import Backend


class Hasher:

    _true_hash = bool(Backend.use_true_hash())

    def __init__(self, seed: int = 0):
        self._seed = seed

    @property
    def uses_true_hash(self) -> bool:
        return self._true_hash

    @staticmethod
    def hash_numpy(x: numpy.ndarray) -> int:
        """Return a value that uniquely identifies a numpy array."""
        x = x.astype("|S576") if x.dtype == "O" else x
        return xxhash.xxh64_hexdigest(x.tobytes())

    @staticmethod
    def hash_torch(x):
        bytes = judo.to_numpy(x).tobytes()
        return xxhash.xxh32_intdigest(bytes)

    @staticmethod
    def get_one_id():
        return uuid.uuid1().int >> 64

    @classmethod
    def true_hash_tensor(cls, x):
        funcs = {
            "numpy": cls.hash_numpy,
            "torch": cls.hash_torch,
        }
        return Backend.execute(x, funcs)

    def hash_tensor(self, x):
        if self._true_hash:
            return self.true_hash_tensor(x)
        return self.get_one_id()

    def hash_iterable(self, x):
        hashes = [self.hash_tensor(xi) for xi in x]
        return judo.as_tensor(hashes, dtype=judo.dtype.hash_type)

    def hash_state(self, state):
        if self._true_hash:
            _hash = hash(
                tuple(
                    [
                        self.hash_tensor(x) if k in state._tensor_names else hash(x)
                        for k, x in state.items()
                    ],
                ),
            )
            return _hash
        return self.get_one_id()


hasher = Hasher()
