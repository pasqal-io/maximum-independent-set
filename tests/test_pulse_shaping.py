import networkx as nx
import pytest

from mis import BackendConfig
from mis.solver.solver import MISInstance
from mis.pipeline.config import SolverConfig
from mis.pipeline.embedder import DefaultEmbedder
from mis.pipeline.pulse import DefaultPulseShaper
from mis._backends.backends import QutipBackend


@pytest.mark.flaky(
    max_runs=5
)  # Layout is non-deterministic, sometimes it will produce weird results.
@pytest.mark.parametrize(
    "size", [3, 5, 6, 7]
)  # No square, the layout produces odd results too often.
def test_pulse_shaping_simple_shapes(size: int) -> None:
    # Create a simple graph with `size` nodes, each of them connected only to the
    # previous and next node.
    graph = nx.Graph()
    edges = []
    for i in range(size):
        edges.append(f"edge {i}")
    graph.add_nodes_from(edges)
    for i in range(len(edges)):
        graph.add_edge(edges[i], edges[(i + 1) % len(edges)])
    instance = MISInstance(graph)

    # Prepare embedder and pulse shaper.
    config = SolverConfig()
    backend = QutipBackend(BackendConfig())
    embedder = DefaultEmbedder()
    shaper = DefaultPulseShaper()

    # Compute parameters
    register = embedder.embed(instance, config, backend)
    parameters = shaper._calculate_parameters(register, backend, instance)

    assert len(parameters.connected) == size
    if size > 3:
        assert len(parameters.disconnected) >= 1

    for connected in parameters.connected:
        for disconnected in parameters.disconnected:
            assert disconnected < connected
