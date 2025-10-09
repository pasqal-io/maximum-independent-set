import networkx as nx
import pytest

from mis import MISInstance


def empty_graph() -> nx.Graph:
    return nx.Graph()


def one_node_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node("solo")
    return graph


def simple_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=15, p=0.3, seed=42)


def complex_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=30, p=0.4, seed=42)


@pytest.fixture
def python_dependency_graph() -> MISInstance:
    """
    Reproducing the example for issues 116, 135, 136.
    """
    PYTHON_VERSIONS = ["Python 3.9", "Python 3.10", "Python 3.11", "Python 3.12", "Python 3.13"]
    graph = nx.Graph()
    graph.add_nodes_from(PYTHON_VERSIONS)
    graph.add_nodes_from(["mygreatlib", "anotherlib"])

    for i, v in enumerate(PYTHON_VERSIONS):
        for w in PYTHON_VERSIONS[i + 1 :]:
            graph.add_edge(v, w)

    graph.add_edge("mygreatlib", "Python 3.11")
    graph.add_edge("mygreatlib", "Python 3.12")
    graph.add_edge("anotherlib", "Python 3.9")
    graph.add_edge("anotherlib", "Python 3.12")
    return MISInstance(graph)
