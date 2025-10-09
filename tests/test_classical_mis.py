from typing import Callable
import networkx as nx
import pytest
from pathlib import Path

# Define classical solver configuration
from mis.data.graphs import load_dimacs
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.shared.types import MethodType
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.maximization import Maximization
from mis.shared.types import Weighting

TEST_DIMACS_FILES_DIR = Path.cwd() / "tests/test_files/dimacs"


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("sample", [(TEST_DIMACS_FILES_DIR / "a265032_1tc.32.txt", 32, 68, 12)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_for_dimacs_32_node_graph(
    sample: tuple[Path, int, int, int],
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    weighting: Weighting,
) -> None:
    """
    Classical MIS solver for a standard graph benchmark dataset in DIMACS format.

    Can be found here: https://oeis.org/A265032/a265032.html
    """
    path, num_nodes, num_edges, mis_size = sample
    dataset = load_dimacs(path)

    assert dataset.instance.graph.number_of_nodes() == num_nodes
    assert dataset.instance.graph.number_of_edges() == num_edges

    config = SolverConfig(
        method=MethodType.EAGER, max_iterations=1, preprocessor=preprocessor, weighting=weighting
    )

    # Run the solver
    solver = MISSolver(dataset.instance, config)
    solutions = solver.solve()

    # Check the solution is genuinely an independent set.
    kernel = Kernelization(config, graph=dataset.instance.original_graph)
    solution = solutions[0]

    # Is it an independent set?
    assert kernel.is_independent(solution.nodes)
    # Is it a subset of the original graph?
    for node in solution.nodes:
        assert node in dataset.instance.node_label_to_index
    # Is it as good as we hope?
    if preprocessor is None:
        assert len(solution.nodes) <= mis_size
    else:
        assert len(solution.nodes) == mis_size


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("sample", [(TEST_DIMACS_FILES_DIR / "a265032_1dc.64.txt", 64, 543, 9)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_for_dimacs_64_node_graph(
    sample: tuple[Path, int, int, int],
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    weighting: Weighting,
) -> None:
    """
    Classical MIS solver for a standard graph benchmark dataset in DIMACS format.

    Can be found here: https://oeis.org/A265032/a265032.html
    """
    path, num_nodes, num_edges, mis_size = sample
    dataset = load_dimacs(path)

    assert dataset.instance.graph.number_of_nodes() == num_nodes
    assert dataset.instance.graph.number_of_edges() == num_edges

    config = SolverConfig(
        method=MethodType.EAGER, max_iterations=1, preprocessor=preprocessor, weighting=weighting
    )

    # Run the solver
    solver = MISSolver(dataset.instance, config)
    solutions = solver.solve()

    # Check the solution is genuinely an independent set.
    kernel = Kernelization(config, graph=dataset.instance.original_graph)

    solution = solutions[0]

    # Is it an independent set?
    assert kernel.is_independent(solution.nodes)
    # Is it a subset of the original graph?
    for node in solution.nodes:
        assert node in dataset.instance.node_label_to_index
    # Is it as good as we hope?
    if preprocessor is None or weighting == Weighting.WEIGHTED:
        # Note: As of this writing, the result we achieve with wMIS is not as good
        # as the result we achieve with regular MIS. See issue #104.
        assert len(solution.nodes) <= mis_size
    else:
        assert len(solution.nodes) == mis_size


@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_empty_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
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
        weighting=weighting,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions) == 0

    with pytest.raises(NotImplementedError):
        solver.embedding()

    with pytest.raises(NotImplementedError):
        solver.pulse(None)


@pytest.mark.parametrize("postprocessor", [None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_disconnected_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
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
        weighting=weighting,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions) == 1
    assert len(solutions[0].nodes) == SIZE


@pytest.mark.parametrize("postprocessor", [None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_star_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
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
        weighting=weighting,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions) == 1
    node_indices = solutions[0].node_indices
    node_set = set(node_indices)
    assert len(node_indices) == SIZE - 1
    assert len(solutions[0].nodes) == len(node_indices)
    assert len(node_indices) == len(node_set)

    for node in node_indices:
        assert 1 <= node < SIZE


@pytest.mark.parametrize(
    "sample",
    [
        (TEST_DIMACS_FILES_DIR / "petersen.txt", 10, 15, 5),
        (TEST_DIMACS_FILES_DIR / "a265032_1dc.64.txt", 64, 543, 10),
        (TEST_DIMACS_FILES_DIR / "a265032_1tc.32.txt", 32, 68, 12),
        (TEST_DIMACS_FILES_DIR / "hexagon.txt", 6, 6, 3),
    ],
)
@pytest.mark.parametrize("postprocessor", [None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_dimacs_mis(
    sample: tuple[Path, int, int, int],
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
) -> None:
    """
    Test loading various graphs from DIMACS files and solving them.
    """
    path, num_nodes, num_edges, mis_size = sample
    dataset = load_dimacs(path)

    assert dataset.instance.graph.number_of_nodes() == num_nodes
    assert dataset.instance.graph.number_of_edges() == num_edges

    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        weighting=weighting,
    )

    solver = MISSolver(dataset.instance, config)
    solutions = solver.solve()

    assert dataset.instance.graph.number_of_nodes() == num_nodes
    assert dataset.instance.graph.number_of_edges() == num_edges
    assert len(solutions) > 0
    for solution in solutions:
        # Is it an independent set?
        kernel = Kernelization(config, dataset.instance.original_graph)
        assert kernel.is_independent(solution.nodes)
        # Is it a subset of the original graph?
        for node in solution.nodes:
            assert node in dataset.instance.node_label_to_index
        assert len(solution.nodes) <= mis_size
