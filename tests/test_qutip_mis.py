import networkx as nx

# Define classical solver configuration
from mis.pipeline.backends import QutipBackend
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.shared.types import MethodType


def test_empty_qtip_mis() -> None:
    """
    Classical MIS solver should work on an empty graph.
    """
    graph = nx.Graph()
    config = SolverConfig(method=MethodType.EAGER, max_iterations=1, backend=QutipBackend())

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 0


def test_disconnected_qtip_mis() -> None:
    """
    Classical MIS solver should work on a graph without any edge.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1, backend=QutipBackend())

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) == SIZE:
            assert len(set(solution.nodes)) == SIZE
            found = True
            break
    assert found


def test_star_qtip_mis() -> None:
    """
    Classical MIS solver should work on a star-shaped graph.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)
        if i != 0:
            graph.add_edge(0, i)

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1, backend=QutipBackend())

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) == SIZE - 1:
            assert len(set(solution.nodes)) == SIZE - 1
            found = True
            break
    assert found
