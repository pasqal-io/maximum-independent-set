from typing import Callable
import networkx as nx
import pytest

from mis import BackendConfig
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.preprocessor import BasePreprocessor
from mis.pipeline.maximization import Maximization
from mis.shared.types import MethodType, Weighting


@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_empty_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], BasePreprocessor],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
) -> None:
    """
    Classical MIS solver should work on an empty graph.
    """
    graph = nx.Graph()
    config = SolverConfig(
        method=MethodType.EAGER,
        max_iterations=1,
        backend=BackendConfig(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        weighting=weighting,
        max_number_of_solutions=10,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions) == 0


@pytest.mark.parametrize(
    "postprocessor",
    argvalues=[
        pytest.param(None, marks=pytest.mark.flaky(max_runs=5)),
        lambda config: Maximization(config),
    ],
)
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_disconnected_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], BasePreprocessor],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
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
        backend=BackendConfig(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        weighting=weighting,
        max_number_of_solutions=10,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    assert solver.embedding() is not None
    assert solver.pulse(solver.embedding()) is not None
    solutions = solver.solve()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) >= SIZE / 2:
            assert len(set(solution.nodes)) == len(solution.nodes)
            found = True
            break
    if preprocessor is not None or postprocessor is not None:
        assert found


@pytest.mark.parametrize(
    "postprocessor",
    argvalues=[
        pytest.param(None, marks=pytest.mark.flaky(max_runs=5)),
        lambda config: Maximization(config),
    ],
)
@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
def test_star_qtip_mis(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], BasePreprocessor],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
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
        backend=BackendConfig(),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        weighting=weighting,
        max_number_of_solutions=10,
    )

    # Create the MIS instance
    instance = MISInstance(graph)

    # Run the solver
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    # Check that at least one of the solutions makes sense.
    found = False
    for solution in solutions:
        assert len(solution.nodes) <= SIZE
        if len(solution.nodes) >= SIZE / 2:
            # Check that nodes are distinct.
            assert len(set(solution.nodes)) == len(solution.nodes)
            found = True
            break
    if preprocessor is not None or postprocessor is not None:
        assert found
