from typing import Callable
import networkx as nx
import pytest
from pathlib import Path

# Define classical solver configuration
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.shared.types import MethodType
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.maximization import Maximization

TEST_DIMACS_FILES_DIR = Path.cwd() / "tests/test_files/dimacs"


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "dimacs_to_nx", [(TEST_DIMACS_FILES_DIR / "a265032_1tc.32.txt", 32, 68, 12)], indirect=True
)
@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
def test_for_dimacs_32_node_graph(
    dimacs_to_nx: tuple[nx.Graph, int, int, int],
    preprocessor: None | Callable[[nx.Graph], Kernelization],
) -> None:
    """
    Classical MIS solver for a standard graph benchmark dataset in DIMACS format.

    Can be found here: https://oeis.org/A265032/a265032.html
    """
    graph, n_nodes, n_edges, mis_size = dimacs_to_nx
    assert graph.number_of_nodes() == n_nodes
    assert graph.number_of_edges() == n_edges

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1, preprocessor=None)

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions[0].nodes) == mis_size

    # Check the solution is genuinely an independent set.
    kernel = Kernelization(graph=graph)
    assert kernel.is_independent(solutions[0].nodes)


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "dimacs_to_nx", [(TEST_DIMACS_FILES_DIR / "a265032_1dc.64.txt", 64, 543, None)], indirect=True
)
@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
def test_for_dimacs_64_node_graph(
    dimacs_to_nx: tuple[nx.Graph, int, int, int],
    preprocessor: None | Callable[[nx.Graph], Kernelization],
) -> None:
    """
    Classical MIS solver for a standard graph benchmark dataset in DIMACS format.

    Can be found here: https://oeis.org/A265032/a265032.html
    """
    graph, n_nodes, n_edges, _ = dimacs_to_nx
    assert graph.number_of_nodes() == n_nodes
    assert graph.number_of_edges() == n_edges

    config = SolverConfig(method=MethodType.EAGER, max_iterations=1, preprocessor=preprocessor)

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    # Check the solution is genuinely an independent set.
    kernel = Kernelization(graph=graph)

    # In this case, pre-processing the data reveals a better solution.
    if preprocessor is None:
        assert set(solutions[0].nodes) == {1, 4, 42, 11, 16, 56, 25, 61}
        assert kernel.is_independent(solutions[0].nodes)
    else:
        assert set(solutions[0].nodes) == {64, 1, 4, 13, 16, 22, 47, 49, 52}
        assert kernel.is_independent(solutions[0].nodes)


@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
@pytest.mark.parametrize("postprocessor", [None, lambda: Maximization()])
def test_empty_mis(
    preprocessor: None | Callable[[nx.Graph], Kernelization],
    postprocessor: None | Callable[[], Maximization],
) -> None:
    """
    Classical MIS solver should work with an empty graph.
    """
    graph = nx.Graph()
    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 0


@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
@pytest.mark.parametrize("postprocessor", [None, lambda: Maximization()])
def test_disconnected_mis(
    preprocessor: None | Callable[[nx.Graph], Kernelization],
    postprocessor: None | Callable[[], Maximization],
) -> None:
    """
    Classical MIS solver should work with a graph without any edge.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)

    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 1
    assert len(solutions[0].nodes) == SIZE


@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
@pytest.mark.parametrize("postprocessor", [None, lambda: Maximization()])
def test_star_mis(
    preprocessor: None | Callable[[nx.Graph], Kernelization],
    postprocessor: None | Callable[[], Maximization],
) -> None:
    """
    Classical MIS solver should work with a star-shaped graph.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)
        if i != 0:
            graph.add_edge(0, i)

    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

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


@pytest.mark.parametrize(
    "dimacs_to_nx",
    [
        (TEST_DIMACS_FILES_DIR / "petersen.txt", 10, 15, 5),
        (TEST_DIMACS_FILES_DIR / "a265032_1dc.64.txt", 64, 543, 10),
        (TEST_DIMACS_FILES_DIR / "a265032_1tc.32.txt", 32, 68, 12),
        (TEST_DIMACS_FILES_DIR / "hexagon.txt", 6, 6, 3),
    ],
    indirect=True,
)
@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
@pytest.mark.parametrize("postprocessor", [None, lambda: Maximization()])
def test_dimacs_mis(
    dimacs_to_nx: tuple[nx.Graph, int, int, int],
    preprocessor: None | Callable[[nx.Graph], Kernelization],
    postprocessor: None | Callable[[], Maximization],
) -> None:
    """
    Test loading various graphs from DIMACS files and solving them.
    """
    graph, num_nodes, num_edges, max_independent_set_size = dimacs_to_nx

    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    instance = MISInstance(graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert graph.number_of_nodes() == num_nodes
    assert graph.number_of_edges() == num_edges
    assert len(solutions) > 0
    for solution in solutions:
        assert Kernelization(graph).is_independent(solution.nodes)
        assert len(solution.nodes) <= max_independent_set_size
