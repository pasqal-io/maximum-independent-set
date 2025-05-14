import abc

import networkx as nx


class BasePreprocessor(abc.ABC):
    """
    Base class for preprocessors.
    """

    @abc.abstractmethod
    def preprocess(self) -> nx.Graph: ...

    """
    Preprocess a graph.

    Typically, this will do two things:

    1. apply transformations to the graph to reduce its size
    2. store rebuild operations.

    If several preprocessors are chained (e.g. call `A.preprocess`,
    then `B.preprocess`), the caller MUST ensure that the `rebuild`
    operations are called in the opposite order from the `preprocess`
    operations (e.g. `B.rebuild` before `A.rebuild`).
    """

    @abc.abstractmethod
    def rebuild(self, partial_solution: set[int]) -> set[int]: ...

    """
    Apply any pending rebuild operations.

    During preprocessing, a preprocessor manipulates the graph,
    typically by collapsing nodes using various strategies. Consequently,
    the solutions obtained on a preprocessed graph are not solutions on
    the original graph. For this reason, any preprocessing step will
    typically store a rebuild operation that uncollapses nodes and
    adapts solutions that work on the graph after node collapse into
    a solution that works on the graph before node collapse.

    If several preprocessors are chained (e.g. call `A.preprocess`,
    then `B.preprocess`), the caller MUST ensure that the `rebuild`
    operations are called in the opposite order from the `preprocess`
    operations (e.g. `B.rebuild` before `A.rebuild`).
    """
