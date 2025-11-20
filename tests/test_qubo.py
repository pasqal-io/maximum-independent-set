import pytest
from conftest import simple_graph
import networkx as nx
from mis.solver.solver import MISInstance


def test_qubo_construction() -> None:

    graph: nx.Graph = simple_graph()
    instance = MISInstance(graph)

    qubo = instance.to_qubo()
    assert qubo.shape[0] == qubo.shape[1] == graph.number_of_nodes()
    assert qubo.max() == 2.5

    with pytest.raises(ValueError):
        instance.to_qubo(penalty=0.1)
