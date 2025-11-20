import pytest
from conftest import simple_graph

import sys
import networkx as nx
from mis.solver.solver import MISInstance


def test_qubo_construction() -> None:

    graph: nx.Graph = simple_graph()
    if sys.version_info[1] < 13:
        instance = MISInstance(graph)

        qubo = instance.to_qubo()
        assert qubo.size == graph.number_of_nodes()
        assert qubo.coefficients.max().item() == 2.5

        with pytest.raises(ValueError):
            instance.to_qubo(penalty=0.1)
