import pytest
import networkx as nx
from typing import Callable

from mis import BackendConfig
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig, GreedyConfig
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.maximization import Maximization
from mis.shared.types import MethodType, Weighting
from mis.shared.graphs import is_independent

from conftest import simple_graph, empty_graph, one_node_graph, complex_graph


@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
@pytest.mark.parametrize("use_quantum", [False, True])
@pytest.mark.parametrize(
    "simple_graph", argvalues=[simple_graph(), empty_graph(), one_node_graph()]
)
def test_greedy_mis_basic(
    simple_graph: nx.Graph,
    use_quantum: bool,
    weighting: Weighting,
) -> None:
    """
    Test Greedy MIS solver in both classical and quantum modes with default settings.
    """
    backend = BackendConfig() if use_quantum else None
    config = SolverConfig(
        method=MethodType.GREEDY,
        backend=backend,
        weighting=weighting,
        greedy=GreedyConfig(),
    )
    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    if len(simple_graph) > 0:
        assert len(solutions) > 0
    assert all(isinstance(sol.nodes, list) for sol in solutions)
    assert all(is_independent(instance.graph, sol.nodes) for sol in solutions)


@pytest.mark.parametrize("preprocessor", [None, lambda config, graph: Kernelization(config, graph)])
@pytest.mark.parametrize("postprocessor", argvalues=[None, lambda config: Maximization(config)])
@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
@pytest.mark.parametrize("use_quantum", [False, True])
@pytest.mark.parametrize(
    "simple_graph", argvalues=[simple_graph(), empty_graph(), one_node_graph()]
)
def test_greedy_solver_with_pre_post(
    preprocessor: None | Callable[[SolverConfig, nx.Graph], Kernelization],
    postprocessor: None | Callable[[SolverConfig], Maximization],
    weighting: Weighting,
    use_quantum: bool,
    simple_graph: nx.Graph,
) -> None:
    """
    Test greedy solver behavior with optional pre- and postprocessing,
    in both classical and quantum modes.
    """

    if use_quantum:
        if not all("pos" in simple_graph.nodes[n] for n in simple_graph.nodes):
            for i, node in enumerate(simple_graph.nodes):
                simple_graph.nodes[node]["pos"] = (i * 1.0, 0.0)

    backend = BackendConfig() if use_quantum else None

    config = SolverConfig(
        method=MethodType.GREEDY,
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        weighting=weighting,
        greedy=GreedyConfig(),
    )

    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    if len(simple_graph) > 0:
        assert len(solutions) > 0
    for solution in solutions:
        assert isinstance(solution.nodes, list)
        assert len(set(solution.nodes)) == len(solution.nodes)
        assert is_independent(instance.graph, solution.nodes)


@pytest.mark.parametrize("weighting", argvalues=[Weighting.UNWEIGHTED, Weighting.WEIGHTED])
@pytest.mark.parametrize("use_quantum", [False, True])
@pytest.mark.parametrize("complex_graph", argvalues=[complex_graph()])
def test_greedy_mis_long(complex_graph: nx.Graph, use_quantum: bool, weighting: Weighting) -> None:
    """
    Test Greedy MIS solver in both classical and quantum modes with default settings.
    """
    backend = BackendConfig() if use_quantum else None
    config = SolverConfig(
        method=MethodType.GREEDY,
        backend=backend,
        weighting=weighting,
        greedy=GreedyConfig(default_solving_threshold=10),
    )
    instance = MISInstance(complex_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve()

    assert len(solutions) > 0
    assert all(isinstance(sol.nodes, list) for sol in solutions)
    assert all(is_independent(instance.graph, sol.nodes) for sol in solutions)

    with pytest.raises(NotImplementedError):
        solver.embedding()

    with pytest.raises(NotImplementedError):
        solver.drive(None)
