from typing import Dict, Generator, Tuple, Union

import numpy

import judo
from judo.typing import Tensor


AVAILABLE_FUNCTIONS = {
    "similiar_chunks_indexes",
    "split_similar_chunks",
    "split_kwargs_in_chunks",
    "split_args_in_chunks",
}


def similiar_chunks_indexes(
    n_values,
    n_chunks,
    allow_size_1: bool = True,
) -> Generator[Tuple[int, int], None, None]:
    """
    Return the indexes for splitting an array in similar chunks.

    Args:
        n_values: Length of the array that will be split.
        n_chunks: Number of similar chunks.
        allow_size_1: If True chunks of size 1 can be returned. If false return \
                     indexes that avoid batches of size 1.

    Returns:
        Generator containing the indexes of every new chunk.

    """
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for i in range(0, n_values, chunk_size):
        if not allow_size_1 and i + chunk_size >= n_values - 2:  # Avoid batches of size 1
            yield i, i + chunk_size + 1
            break
        else:
            yield i, i + chunk_size


def split_similar_chunks(
    vector: Union[list, Tensor],
    n_chunks: int,
    allow_size_1: bool = True,
) -> Generator[Union[list, Tensor], None, None]:
    """
    Split an indexable object into similar chunks.

    Args:
        vector: Target object to be split.
        n_chunks: Number of similar chunks.
        allow_size_1: If True chunks of size 1 can be returned. If false return \
                     indexes that avoid batches of size 1.

    Returns:
        Generator that returns the chunks created after splitting the target object.

    """
    for start, end in similiar_chunks_indexes(len(vector), n_chunks, allow_size_1=allow_size_1):
        yield vector[start:end]


def split_kwargs_in_chunks(
    kwargs: Dict[str, Union[list, Tensor]],
    n_chunks: int,
    allow_size_1: bool = True,
) -> Generator[Dict[str, Union[list, Tensor]], None, None]:
    """Split the kwargs passed to ``make_transitions`` in similar batches."""
    n_values = len(next(iter(kwargs.values())))  # Assumes all data have the same len
    chunk_size = int(numpy.ceil(n_values / n_chunks))
    for start, end in similiar_chunks_indexes(n_values, n_chunks, allow_size_1=allow_size_1):
        if start + chunk_size >= n_values - 2:  # Do not allow the last chunk to have size 1
            yield {k: v[start:n_values] if judo.is_tensor(v) else v for k, v in kwargs.items()}
            break
        else:
            yield {k: v[start:end] if judo.is_tensor(v) else v for k, v in kwargs.items()}


def split_args_in_chunks(
    args: Tuple[Union[list, Tensor]],
    n_chunks: int,
    allow_size_1: bool = True,
) -> Generator[Union[list, Tensor], None, None]:
    """Split the args passed to ``make_transitions`` in similar batches."""
    n_values = len(args[0])
    for start, end in similiar_chunks_indexes(n_values, n_chunks, allow_size_1=allow_size_1):
        yield tuple(v[start:end] for v in args)
