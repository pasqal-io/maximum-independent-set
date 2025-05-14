import pytest

import networkx as nx


@pytest.fixture
def dimacs_to_nx(request: pytest.FixtureRequest) -> nx.Graph:
    graph = nx.Graph()
    with open(request.param, "r") as f:
        for line in f:
            if line.startswith("c"):  # Comment line in DIMACS file.
                continue
            elif line.startswith("p"):  # Problem definition, i.e. # nodes and edges.
                _, _, num_nodes, num_edges = line.strip().split()
            elif line.startswith("e"):
                _, node1, node2 = line.strip().split()
                graph.add_edge(int(node1), int(node2))
    return graph
