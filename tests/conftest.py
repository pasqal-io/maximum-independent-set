import pytest

import networkx as nx


@pytest.fixture
def simple_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=15, p=0.3, seed=42)


@pytest.fixture
def complex_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=30, p=0.4, seed=42)
