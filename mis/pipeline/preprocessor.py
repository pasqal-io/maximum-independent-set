import abc

import networkx as nx


class BasePreprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, graph: nx.Graph) -> nx.Graph: ...

    @abc.abstractmethod
    def rebuild(self, partial_solution: set[int]) -> set[int]: ...
