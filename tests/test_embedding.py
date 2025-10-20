import pytest
import networkx as nx
from mis import BackendConfig
from mis.pipeline.config import SolverConfig
from mis.pipeline.embedder import DefaultEmbedder, OptimizedEmbedder
from mis.solver.solver import MISInstance, MISSolver, MISSolverQuantum
from conftest import simple_graph, dimacs_16nodes


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "graph, default_embedder_fails", [(simple_graph(), False), (dimacs_16nodes(), True)]
)
def test_easy_embedding(graph: nx.Graph, default_embedder_fails: bool) -> None:

    instance = MISInstance(graph)
    config = SolverConfig(preprocessor=None, backend=BackendConfig())
    solver = MISSolver(instance, config)
    assert isinstance(solver._solver, MISSolverQuantum)
    assert isinstance(solver._solver._embedder, DefaultEmbedder)  # type: ignore[attr-defined]

    register = solver.embedding()
    assert len(register.qubits) == len(graph)
    if default_embedder_fails:
        with pytest.raises(Exception):
            solver._solver.backend.device().validate_register(register)
    else:
        assert solver._solver.backend.device().validate_register(register) is None

    opt_config = SolverConfig(
        preprocessor=None, backend=BackendConfig(), embedder=OptimizedEmbedder()
    )
    opt_solver = MISSolver(instance, opt_config)
    assert isinstance(solver._solver, MISSolverQuantum)
    assert isinstance(opt_solver._solver._embedder, OptimizedEmbedder)  # type: ignore[attr-defined]
    register = opt_solver.embedding()
    assert len(register.qubits) == len(graph)
    assert opt_solver._solver.backend.device().validate_register(register) is None  # type: ignore[attr-defined]
