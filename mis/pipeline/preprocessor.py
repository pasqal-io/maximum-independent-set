import abc

import networkx as nx


class BasePreprocessor(abc.ABC):
    """
    Base class for preprocessors.
    """

    @abc.abstractmethod
    def preprocess(self, graph: nx.Graph) -> nx.Graph: ...

    """
    Preprocess a graph.

    Typically, this will do two things:

    1. apply transformations to the graph to reduce its size
    2. store rebuild operations that convert solutions on the reduced
        graph into solutions on the original graph.

    If several preprocessors are chained (e.g. call `A.preprocess`,
    then `B.preprocess`), the caller MUST ensure that the `rebuild`
    operations are called in the opposite order from the `preprocess`
    operations (e.g. `B.rebuild` before `A.rebuild`).
    """

    @abc.abstractmethod
    def rebuild(self, partial_solution: set[int]) -> set[int]: ...

    """
    Apply any pending rebuild operations.
    """


class EmptyPreprocessor(BasePreprocessor):
    """
    A preprocessor that does nothing.
    """

    def preprocess(self, graph: nx.Graph) -> nx.Graph:
        return graph

    def rebuild(self, partial_solution: set[int]) -> set[int]:
        return partial_solution
