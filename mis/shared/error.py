"""
Exceptions raised within this library.
"""


class GraphError(ValueError):
    """
    An error raised when attempting to compile a graph, for any reason.
    """


class MappingError(ValueError):
    """
    An error raised when a graph-to-layout mapping fails validation,
    such as when node placements violate adjacency or physical layout constraints.
    """


class CompilationError(GraphError):
    """
    An error raised when attempting to compile a graph for an architecture
    that does not support it, e.g. because it requires too many qubits or
    because the physical constraints on the geometry are not satisfied.
    """

    pass


class ExecutionError(Exception):
    pass
