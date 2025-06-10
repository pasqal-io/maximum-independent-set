from __future__ import annotations

import networkx as nx


def calculate_weight(graph: nx.Graph, nodes: list[int]) -> float:
    """
    Calculates the total weight of a set of nodes in a given MISInstance

    Args:
        graph: The graph to check.
        nodes: List of node indices.

    Returns:
        Total weight as a float.
    """
    return float(sum(graph.nodes[n].get("weight", 1.0) for n in nodes))


def is_independent(graph: nx.Graph, nodes: list[int]) -> bool:
    """
    Checks if the node set is an independent set (no edges between them).

    Args:
        graph: The graph to check.
        nodes: The set of nodes.

    Returns:
        True if independent, False otherwise.
    """
    return not any(graph.has_edge(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :])


def remove_neighborhood(graph: nx.Graph, nodes: list[int]) -> nx.Graph:
    """
    Removes a node and all its neighbors from the graph.

    Args:
        graph: The graph to modify.
        nodes: List of nodes to remove.

    Returns:
        The reduced graph.
    """
    reduced = graph.copy()
    to_remove = set(nodes)
    for node in nodes:
        to_remove.update(graph.neighbors(node))
    reduced.remove_nodes_from(to_remove)
    return reduced
