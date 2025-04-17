import networkx as nx

# Define classical solver configuration
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.shared.types import MethodType


def test_empty_mis() -> None:
    """
    Classical MIS solver should work with an empty graph.
    """
    graph = nx.Graph()
    config = SolverConfig(method=MethodType.EAGER, max_iterations=1)

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 0


def test_disconnected_mis() -> None:
    """
    Classical MIS solver should work with a graph without any edge.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1)

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 1
    assert len(solutions[0].nodes) == SIZE


def test_star_mis() -> None:
    """
    Classical MIS solver should work with a star-shaped graph.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)
        if i != 0:
            graph.add_edge(0, i)

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1)

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 1
    nodes = solutions[0].nodes
    node_set = set(nodes)
    assert len(nodes) == SIZE - 1
    assert len(nodes) == len(node_set)

    for node in nodes:
        assert 1 <= node < SIZE
