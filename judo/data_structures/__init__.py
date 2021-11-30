from judo.data_structures.bounds import Bounds
from judo.data_structures.memory import ReplayMemory
from judo.data_structures.states import States


try:
    from judo.data_structures.tree import HistoryTree
except ImportError:
    HistoryTree = None
