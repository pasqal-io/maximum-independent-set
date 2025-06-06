import pytest
import networkx as nx
from typing import Callable

from mis.pipeline.backends import QutipBackend
from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig, GreedyConfig
from mis.pipeline.kernelization import Kernelization
from mis.pipeline.maximization import Maximization
from mis.shared.types import MethodType
from mis.shared.graphs import is_independent


@pytest.mark.parametrize("use_quantum", [False, True])
def test_greedy_mis_basic(simple_graph: nx.Graph, use_quantum: bool) -> None:
    """
    Test Greedy MIS solver in both classical and quantum modes with default settings.
    """
    backend = QutipBackend() if use_quantum else None
    config = SolverConfig(
        method=MethodType.GREEDY, use_quantum=use_quantum, backend=backend, greedy=GreedyConfig()
    )
    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) > 0
    assert all(isinstance(sol.nodes, list) for sol in solutions)
    assert all(is_independent(instance.graph, sol.nodes) for sol in solutions)


@pytest.mark.parametrize("preprocessor", [None, lambda graph: Kernelization(graph)])
@pytest.mark.parametrize("postprocessor", [None, lambda: Maximization()])
@pytest.mark.parametrize("use_quantum", [False, True])
def test_greedy_solver_with_pre_post(
    preprocessor: None | Callable[[nx.Graph], Kernelization],
    postprocessor: None | Callable[[], Maximization],
    use_quantum: bool,
    simple_graph: nx.Graph,
) -> None:
    """
    Test greedy solver behavior with optional pre- and postprocessing,
    in both classical and quantum modes.
    """
    # TODO: FIX greedy algorithm without preprocessing.
    # Needs to be investigated. Possibly because of misalignment in the node_ids of the
    # preprocessed graph, and the subgraph built on the layout in the greedy algorithm
    if preprocessor is None and use_quantum:
        pytest.skip("Skipping test because postprocessor is None.")

    if use_quantum:
        if not all("pos" in simple_graph.nodes[n] for n in simple_graph.nodes):
            for i, node in enumerate(simple_graph.nodes):
                simple_graph.nodes[node]["pos"] = (i * 1.0, 0.0)

    backend = QutipBackend() if use_quantum else None

    config = SolverConfig(
        method=MethodType.GREEDY,
        use_quantum=use_quantum,
        backend=backend,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        greedy=GreedyConfig(),
    )

    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) > 0
    for solution in solutions:
        assert isinstance(solution.nodes, list)
        assert len(set(solution.nodes)) == len(solution.nodes)
        assert is_independent(instance.graph, solution.nodes)
