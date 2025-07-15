from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import L
from .types import Objective

import networkx as nx


class BaseCostPicker(ABC):
    @abstractmethod
    @classmethod
    def node_weight(cls, node: dict[str, float]) -> float:
        """
        Get the weight of a node.

        For a weighted cost picker, this returns attribute `weight` of the node,
        or 1. if the node doesn't specify a `weight`.

        For an unweighted cost picker, this always returns 1.
        """
        ...

    @abstractmethod
    @classmethod
    def set_node_weight(cls, node: dict[str, float], weight: float):
        """
        Set the weight of a node.

        For a weighted cost picker, this returns attribute `weight` of the node,
        or 1. if the node doesn't specify a `weight`.

        For an unweighted cost picker, raise an error.
        """
        ...

    @abstractmethod
    @classmethod
    def node_delta(cls, node: dict[str, float], delta: float) -> float:
        """
        Apply a delta to the weight of a node.

        Raises an error in an unweighted cost picker.
        """
        ...

    @abstractmethod
    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        """
        Get the weight of a subraph.

        See `node_weight` for the definition of weight.

        For an unweighted cost picker, this always returns `len(nodes)`.
        """
        ...

    @classmethod
    def for_objective(cls, objective: Objective) -> type[BaseCostPicker]:
        """
        Pick a cost picker for an objective.
        """
        if objective == Objective.MAXIMIZE_SIZE:
            return UnweightedPicker
        elif objective == Objective.MAXIMIZE_WEIGHT:
            return WeightedPicker


class WeightedPicker(BaseCostPicker):
    @classmethod
    def node_weight(cls, node: dict[str, float]) -> float:
        return node.get("weight", 1.0)

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(sum(cls.node_weight(graph.nodes[n]) for n in nodes))

class UnweightedPicker(BaseCostPicker):
    @classmethod
    def node_weight(cls, node: dict[str, float]) -> float:
        return 1.0

    @classmethod
    def node_delta(cls, node: dict[str, float], delta: float) -> float:
        raise NotImplementedError("UnweightedPicker does not support operation `node_delta`")

    @classmethod
    def set_node_weight(cls, node: dict[str, float], weight: float):
        raise NotImplementedError("UnweightedPicker does not support operation `set_node_weight`")

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(len(nodes))

@dataclass
class ClosedNeighborhood:
    original_node: int
    graph: nx.Graph
    nodes: list[int]

    def is_isolated(self) -> bool:
        for i, u in enumerate(self.nodes):
            for v in self.nodes[i + 1:]:
                if self.graph.has_edge(u, v):
                    return False
        return True


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
