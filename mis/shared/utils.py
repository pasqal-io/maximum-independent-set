from __future__ import annotations

import networkx as nx

from mis.shared.types import MISInstance


def calculate_weight(instance: MISInstance, nodes: list[int]) -> float:
    """
    Calculates the total weight of a set of nodes in a given MISInstance

    Args:
        nodes: List of node indices.

    Returns:
        Total weight as a float.
    """
    return sum(instance.graph.nodes[n].get("weight", 1.0) for n in nodes)

def is_independent(graph: nx.Graph, nodes: list[int]) -> bool:
    """
    Checks if the node set is an independent set (no edges between them).

    Args:
        graph: The graph to check.
        nodes: The set of nodes.

    Returns:
        True if independent, False otherwise.
    """
    return not any(graph.has_edge(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:])