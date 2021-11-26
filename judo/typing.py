from typing import Any, Dict, Generator, List, Set, Tuple, Union

import numpy

from judo.judo_backend import torch


StateDict = Dict[str, Dict[str, Any]]

Tensor = Union[numpy.ndarray, torch.Tensor]
Vector = Union[numpy.ndarray, torch.Tensor]
Matrix = Union[numpy.ndarray, torch.Tensor]
Scalar = Union[int, float]

NodeId = Union[str, int]
NodeData = Union[Tuple[dict, dict], Tuple[dict, dict, dict]]
NodeDataGenerator = Generator[NodeData, None, None]
NamesData = Union[Tuple[str], Set[str], List[str]]
