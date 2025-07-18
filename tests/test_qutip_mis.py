from typing import Callable
import networkx as nx
import pytest

# Define classical solver configuration
from mis.pipeline.backends import QutipBackend
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.maximization import Maximization
from mis.shared.types import MethodType, Objective


@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize(
    "objective", argvalues=[Objective.MAXIMIZE_SIZE, Objective.MAXIMIZE_WEIGHT]
)
def test_empty_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    objective: Objective,
) -> None:
    """
    Classical MIS solver should work on an empty graph.
    """
    graph = nx.Graph()
    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        backend=QutipBackend(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        objective=objective,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) == 0


@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize(
    "objective", argvalues=[Objective.MAXIMIZE_SIZE, Objective.MAXIMIZE_WEIGHT]
)
def test_disconnected_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    objective: Objective,
) -> None:
    """
    Classical MIS solver should work on a graph without any edge.
    """
    SIZE = 10
    graph = nx.Graph()
    for i in range(SIZE):
        graph.add_node(i)

    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        backend=QutipBackend(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        objective=objective,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) >= SIZE / 2:
            assert len(set(solution.nodes)) == len(solution.nodes)
            found = True
            break
    assert found


@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize(
    "objective", argvalues=[Objective.MAXIMIZE_SIZE, Objective.MAXIMIZE_WEIGHT]
)
def test_star_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    objective: Objective,
) -> None:
    """
    Classical MIS solver should work on a star-shaped graph.
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
        backend=QutipBackend(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        objective=objective,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) >= SIZE / 2:
            # Check that nodes are distinct.
            assert len(set(solution.nodes)) == len(solution.nodes)
            found = True
            break
    assert found
