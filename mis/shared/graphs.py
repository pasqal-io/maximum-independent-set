from __future__ import annotations
from abc import ABC, abstractmethod
from .types import Objective

import networkx as nx


class BaseCostPicker(ABC):
    @abstractmethod
    @classmethod
    def from_node(cls, node: dict[str, float]) -> float:
        ...

    @abstractmethod
    @classmethod
    def from_subgraph(cls, graph: nx.Graph, nodes: list[int]) -> float:
        ...

    @classmethod
    def for_objective(cls, objective: Objective) -> type[BaseCostPicker]:
        if objective == Objective.MAXIMIZE_SIZE:
            return UnweightedPicker
        elif objective == Objective.MAXIMIZE_WEIGHT:
            return WeightPicker


class WeightPicker(BaseCostPicker):
    @classmethod
    def from_node(cls, node: dict[str, float]) -> float:
        return node.get("weight", 1.0)

    @classmethod
    def from_subgraph(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(sum(cls.from_node(graph.nodes[n]) for n in nodes))

class UnweightedPicker(BaseCostPicker):
    @classmethod
    def from_node(cls, node: dict[str, float]) -> float:
        return 1.0

    @classmethod
    def from_subgraph(cls, graph: nx.Graph, nodes: list[int]) -> float:
        return float(len(nodes))


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
