import networkx as nx

from mis import SolverConfig, MISInstance
from mis.pipeline.maximization import Maximization
from mis.shared.types import MISSolution


def python_dependency_graph() -> MISInstance:
    """
    Reproducing the example for issue 116.
    """
    PYTHON_VERSIONS = ["Python 3.9", "Python 3.10", "Python 3.11", "Python 3.12", "Python 3.13"]
    graph = nx.Graph()
    graph.add_nodes_from(PYTHON_VERSIONS)
    graph.add_nodes_from(["mygreatlib", "anotherlib"])

    for i, v in enumerate(PYTHON_VERSIONS):
        for w in PYTHON_VERSIONS[i + 1 :]:
            graph.add_edge(v, w)

    graph.add_edge("mygreatlib", "Python 3.11")
    graph.add_edge("mygreatlib", "Python 3.12")
    graph.add_edge("anotherlib", "Python 3.9")
    graph.add_edge("anotherlib", "Python 3.12")
    return MISInstance(graph)


def test_independence() -> None:
    """Testing that independent solutions are marked as independent and dependent solutions are turned independent"""
    instance = python_dependency_graph()
    maximizer = Maximization(SolverConfig())

    incomplete_solution = MISSolution(
        instance, instance.node_indices(["mygreatlib", "anotherlib"]), frequency=0.5
    )
    complete_solution = MISSolution(
        instance, instance.node_indices(["mygreatlib", "anotherlib", "Python 3.10"]), frequency=0.5
    )
    complete_plus_conflict_solution = MISSolution(
        instance,
        instance.node_indices(["mygreatlib", "anotherlib", "Python 3.10", "Python 3.11"]),
        frequency=0.5,
    )
    conflict_solution = MISSolution(
        instance, instance.node_indices(["mygreatlib", "anotherlib", "Python 3.11"]), frequency=0.5
    )

    # Check that known-to-be-independent solutions are marked as independent.
    for candidate in incomplete_solution, complete_solution:
        assert maximizer.is_independent_solution(candidate)

    # Check that known-to-be-dependent solutions are marked as dependent.
    for candidate in complete_plus_conflict_solution, conflict_solution:
        assert not maximizer.is_independent_solution(candidate)

    complete_plus_conflict_reduced = maximizer.reduce_to_independence(
        complete_plus_conflict_solution
    )
    for node in complete_plus_conflict_reduced.node_indices:
        assert node in complete_plus_conflict_solution.node_indices
    assert maximizer.is_independent_solution(complete_plus_conflict_reduced)
    assert nx.is_dominating_set(instance.graph, complete_plus_conflict_reduced.node_indices)

    conflict_reduced = maximizer.reduce_to_independence(conflict_solution)
    for node in conflict_reduced.node_indices:
        assert node in conflict_solution.node_indices
    assert maximizer.is_independent_solution(conflict_reduced)
    assert len(conflict_reduced.node_indices) == 2
