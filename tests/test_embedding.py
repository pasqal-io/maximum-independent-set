import pytest
import networkx as nx
from pulser import Register as PulserRegister
from mis.pipeline.config import SolverConfig, LocalEmulator
from mis.pipeline.embedder import DefaultEmbedder, OptimizedEmbedder
from mis.solver.solver import MISInstance, MISSolver, MISSolverQuantum
from conftest import simple_graph, dimacs_16nodes


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "graph, default_embedder_fails", [(simple_graph(), False), (dimacs_16nodes(), True)]
)
def test_easy_embedding(graph: nx.Graph, default_embedder_fails: bool) -> None:

    instance = MISInstance(graph)
    config = SolverConfig(preprocessor=None, backend=LocalEmulator())
    solver = MISSolver(instance, config)
    assert isinstance(solver._solver, MISSolverQuantum)
    assert isinstance(solver._solver._embedder, DefaultEmbedder)  # type: ignore[attr-defined]

    register = solver.embedding()
    assert len(register.qubits) == len(graph)

    conversion_factor = config.device.converter.factors[2]
    pulser_register = PulserRegister(
        {k: pos * conversion_factor for k, pos in register.qubits.items()}
    )
    if default_embedder_fails:
        with pytest.raises(Exception):
            config.device._device.validate_register(pulser_register)
    else:
        assert config.device._device.validate_register(pulser_register) is None

    opt_config = SolverConfig(
        preprocessor=None, backend=LocalEmulator(), embedder=OptimizedEmbedder()
    )
    opt_solver = MISSolver(instance, opt_config)
    assert isinstance(solver._solver, MISSolverQuantum)
    assert isinstance(opt_solver._solver._embedder, OptimizedEmbedder)  # type: ignore[attr-defined]
    register = opt_solver.embedding()
    assert len(register.qubits) == len(graph)

    conversion_factor = config.device.converter.factors[2]
    pulser_register = PulserRegister(
        {k: pos * conversion_factor for k, pos in register.qubits.items()}
    )

    assert opt_config.device._device.validate_register(pulser_register) is None  # type: ignore[attr-defined]
