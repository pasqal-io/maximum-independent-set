from __future__ import annotations
from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    from .types import Weighting

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
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def set_node_weight(cls, graph: nx.Graph, node: int, weight: float) -> None:
        """
        Set the weight of a node.

        For a weighted cost picker, this returns attribute `weight` of the node,
        or 1. if the node doesn't specify a `weight`.

        For an unweighted cost picker, raise an error.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def node_delta(cls, graph: nx.Graph, node: int, delta: float) -> float:
        """
        Apply a delta to the weight of a node.

        Raises an error in an unweighted cost picker.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: typing.Iterable[int]) -> float:
        """
        Get the weight of a subraph.

        See `node_weight` for the definition of weight.

        For an unweighted cost picker, this always returns `len(nodes)`.
        """
        raise NotImplementedError

    @classmethod
    def for_weighting(cls, weighting: Weighting) -> type[BaseWeightPicker]:
        """
        Pick a cost picker for an objective.
        """
        from .types import Weighting

        if weighting == Weighting.UNWEIGHTED:
            return UnweightedPicker
        elif weighting == Weighting.WEIGHTED:
            return WeightedPicker


class WeightedPicker(BaseWeightPicker):
    @classmethod
    def node_weight(cls, graph: nx.Graph, node: int) -> float:
        result = graph.nodes[node].get("weight", 1.0)
        assert isinstance(result, float)
        return result

    @classmethod
    def set_node_weight(cls, graph: nx.Graph, node: int, weight: float) -> None:
        graph.nodes[node]["weight"] = weight

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: typing.Iterable[int]) -> float:
        return float(sum(cls.node_weight(graph, n) for n in nodes))


class UnweightedPicker(BaseWeightPicker):
    @classmethod
    def node_weight(cls, graph: nx.Graph, node: int) -> float:
        return 1.0

    @classmethod
    def set_node_weight(cls, graph: nx.Graph, node: int, weight: float) -> None:
        raise NotImplementedError("UnweightedPicker does not support operation `set_node_weight`")

    @classmethod
    def subgraph_weight(cls, graph: nx.Graph, nodes: typing.Iterable[int]) -> float:
        # In the unweighted picker, we can usually run this function in constant time.
        if hasattr(nodes, "__len__"):
            # Usually, we call this with a list or a set, so `len()` is fast.
            sized = typing.cast(typing.Sized, nodes)
            return float(len(sized))
        # Otherwise, we have to count the number of nodes.
        # Apparently, constructor `tuple` is optimized for this, and clearly faster
        # than calling `node_weight` for each node.
        return float(len(tuple(nodes)))


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
