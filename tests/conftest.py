import networkx as nx


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
