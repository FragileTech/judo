from judo import functions
from judo.judo_backend import Backend


AVAILABLE_FUNCTIONS = set(
    [
        "argmax",
        "hash_numpy",
        "hash_tensor",
        "concatenate",
        "stack",
        "clip",
        "repeat",
        "min",
        "max",
        "norm",
        "unsqueeze",
        "where",
        "sqrt",
        "tile",
        "logical_or",
        "logical_and",
    ]
    + list(functions.batching.AVAILABLE_FUNCTIONS)
    + list(functions.fractalai.AVAILABLE_FUNCTIONS)
    + list(functions.images.AVAILABLE_FUNCTIONS)
    + list(functions.notebook.AVAILABLE_FUNCTIONS),
)


class MetaAPI(type):
    def __getattr__(self, item):
        return self.get_function(name=item)

    @staticmethod
    def get_function(name):
        if name in functions.fractalai.AVAILABLE_FUNCTIONS:
            return getattr(functions.fractalai, name)
        elif name in functions.batching.AVAILABLE_FUNCTIONS:
            return getattr(functions.batching, name)
        elif name in functions.images.AVAILABLE_FUNCTIONS:
            return getattr(functions.images, name)
        elif name in functions.notebook.AVAILABLE_FUNCTIONS:
            return getattr(functions.notebook, name)
        elif Backend.is_numpy():
            backend = functions.numpy
        else:
            backend = functions.pytorch
        return getattr(backend, name)


class API(metaclass=MetaAPI):
    pass
