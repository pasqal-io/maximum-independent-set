import pytest

import networkx as nx
from pathlib import Path
from mis.data.graphs import load_dimacs


@pytest.fixture
def dimacs_to_nx(request: pytest.FixtureRequest) -> tuple[nx.Graph, int, int, int]:
    file_path, num_nodes, num_edges, max_independent_set_size = request.param
    dataset = load_dimacs(Path(file_path))
    graph = dataset.instance.graph
    return graph, num_nodes, num_edges, max_independent_set_size


@pytest.fixture
def simple_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=15, p=0.3, seed=42)
