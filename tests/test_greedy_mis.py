import pytest
import networkx as nx

from mis.solver.solver import MISInstance, MISSolver
from mis.pipeline.config import SolverConfig, GreedyConfig
from mis.pipeline.backends import QutipBackend
from mis.shared.types import MethodType
from mis.shared.utils import is_independent


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


def test_greedy_mis_with_layout_and_blockade(
    simple_graph: nx.Graph, layout_coords: dict[int, tuple[float, float]]
) -> None:
    """
    Test Greedy MIS with custom layout coordinates and rydberg blockade.
    """
    config = SolverConfig(
        method=MethodType.GREEDY,
        greedy=GreedyConfig(layout_coords=layout_coords, rydberg_blockade=5.0),
    )
    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) > 0
    assert all(isinstance(sol.nodes, list) for sol in solutions)
    assert all(is_independent(instance.graph, sol.nodes) for sol in solutions)


def test_quantum_greedy_with_layout(
    simple_graph: nx.Graph, layout_coords: dict[int, tuple[float, float]]
) -> None:
    """
    Test quantum backend + greedy solver with layout coordinates.
    """
    config = SolverConfig(
        method=MethodType.GREEDY,
        use_quantum=True,
        backend=QutipBackend(),
        greedy=GreedyConfig(layout_coords=layout_coords, rydberg_blockade=5.0),
    )
    instance = MISInstance(simple_graph)
    solver = MISSolver(instance, config)
    solutions = solver.solve().result()

    assert len(solutions) > 0
    assert all(isinstance(sol.nodes, list) for sol in solutions)
    assert all(len(set(sol.nodes)) == len(sol.nodes) for sol in solutions)
    assert all(is_independent(instance.graph, sol.nodes) for sol in solutions)
