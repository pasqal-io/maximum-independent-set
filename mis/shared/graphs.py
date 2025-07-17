from __future__ import annotations
from abc import ABC, abstractmethod
from .types import Objective

import networkx as nx


class BaseWeightPicker(ABC):
    """
    Utility class to pick the weight of a node.

    Unweighted implementations optimize the methods into trivial
    operations.
    """

    @classmethod
    @abstractmethod
    def node_weight(cls, graph: nx.Graph, node: int) -> float:
        """
        Get the weight of a node.

        For a weighted cost picker, this returns attribute `weight` of the node,
        or 1. if the node doesn't specify a `weight`.

        For an unweighted cost picker, this always returns 1.
        """
        ...

    @classmethod
    @abstractmethod
    def set_node_weight(cls, graph: nx.Graph, node: int, weight: float):
        """
        Set the weight of a node.

        For a weighted cost picker, this returns attribute `weight` of the node,
        or 1. if the node doesn't specify a `weight`.

        For an unweighted cost picker, raise an error.
        """
        ...

    @classmethod
    @abstractmethod
    def node_delta(cls, graph: nx.Graph, node: int, delta: float) -> float:
        """
        Apply a delta to the weight of a node.

        Raises an error in an unweighted cost picker.
        """
        ...

    @classmethod
    @abstractmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        """
        Get the weight of a subraph.

        See `node_weight` for the definition of weight.

        For an unweighted cost picker, this always returns `len(nodes)`.
        """
        ...

    @classmethod
    def for_objective(cls, objective: Objective) -> type[BaseWeightPicker]:
        """
        Pick a cost picker for an objective.
        """
        if objective == Objective.MAXIMIZE_SIZE:
            return UnweightedPicker
        elif objective == Objective.MAXIMIZE_WEIGHT:
            return WeightedPicker

class WeightedPicker(BaseWeightPicker):
    @classmethod
    def node_weight(cls, graph: nx.Graph, node: int) -> float:
        return graph.nodes[node].get("weight", 1.0)

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(sum(cls.node_weight(graph, n) for n in nodes))

class UnweightedPicker(BaseWeightPicker):
    @classmethod
    def node_weight(cls, graph: nx.Graph, node: int) -> float:
        return 1.0

    @classmethod
    def set_node_weight(cls, graph: nx.Graph, node: int, weight: float):
        raise NotImplementedError("UnweightedPicker does not support operation `set_node_weight`")

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(len(nodes))


def closed_neighborhood(graph: nx.Graph, node: int) -> list[int]:
    """
    Return the list of closed neighbours of a node.
    """
    neighbours = list(graph.neighbors(node))
    neighbours.append(node)
    return neighbours

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
