from __future__ import annotations

from enum import Enum
from typing import Any
import networkx
import matplotlib.pyplot as plt
from mis.shared.graphs import calculate_weight


class MethodType(str, Enum):
    EAGER = "eager"
    GREEDY = "greedy"


class MISInstance:
    def __init__(self, graph: networkx.Graph):
        # FIXME: Make it work with pytorch geometric

        self.original_graph = graph.copy()

        # Our algorithms depend on nodes being consecutive integers, starting
        # from 0, so we first need to rename nodes in the graph.
        self.index_to_node_label: dict[int, Any] = dict()
        self.node_label_to_index: dict[Any, int] = dict()
        for index, label in enumerate(graph.nodes()):
            self.index_to_node_label[index] = label
            self.node_label_to_index[label] = index

        # Copy nodes (and weights), if they exist.
        self.graph = networkx.Graph()
        for index, node in enumerate(graph.nodes()):
            self.graph.add_node(index)
            if "weight" in graph.nodes[node]:
                self.graph.nodes[index]["weight"] = graph.nodes[node]["weight"]

        # Copy edges.
        for u, v in graph.edges():
            index_u = self.node_label_to_index[u]
            index_v = self.node_label_to_index[v]
            self.graph.add_edge(index_u, index_v)

    def draw(
        self,
        nodes: list[int] | None = None,
        node_size: int = 600,
        highlight_color: str = "darkgreen",
        font_family: str = "Century Gothic",
    ) -> None:
        """
        Draw instance graph with highlighted nodes.

        Parameters:

            nodes (list[int]): List of nodes to highlight.
            node_size (int): Size of drawn nodes in drawn graph. (default: 600)
            highlight_color (str): Color to highlight nodes with. (default: "darkgreen")
        """
        # Obtain a view of all nodes
        all_nodes = self.original_graph.nodes
        # Compute graph layout
        node_positions = networkx.kamada_kawai_layout(self.original_graph)
        # Keyword dictionaries to customize appearance
        highlighted_node_kwds = {"node_color": highlight_color, "node_size": node_size}
        unhighlighted_node_kwds = {
            "node_color": "white",
            "edgecolors": "black",
            "node_size": node_size,
        }
        if nodes:  # If nodes is not empty
            original_nodes = [self.index_to_node_label[i] for i in nodes]
            nodeset = set(original_nodes)  # Create a set from node list for easier operations
            if not nodeset.issubset(all_nodes):
                invalid_nodes = list(nodeset - all_nodes)
                bad_nodes = "[" + ", ".join([str(node) for node in invalid_nodes[:10]])
                if len(invalid_nodes) > 10:
                    bad_nodes += ", ...]"
                else:
                    bad_nodes += "]"
                if len(invalid_nodes) == 1:
                    raise Exception("node " + bad_nodes + " is not present in the problem instance")
                else:
                    raise Exception(
                        "nodes " + bad_nodes + " are not present in the problem instance"
                    )
            nodes_complement = all_nodes - nodeset
            # Draw highlighted nodes
            networkx.draw_networkx_nodes(
                self.original_graph,
                node_positions,
                nodelist=original_nodes,
                **highlighted_node_kwds,
            )
            # Draw unhighlighted nodes
            networkx.draw_networkx_nodes(
                self.original_graph,
                node_positions,
                nodelist=list(nodes_complement),
                **unhighlighted_node_kwds,
            )
        else:
            networkx.draw_networkx_nodes(
                self.original_graph,
                node_positions,
                nodelist=list(all_nodes),
                **unhighlighted_node_kwds,
            )
        # Draw node labels
        networkx.draw_networkx_labels(self.original_graph, node_positions, font_family=font_family)
        # Draw edges
        networkx.draw_networkx_edges(self.original_graph, node_positions)
        plt.tight_layout()
        plt.axis("off")
        plt.show()


class MISSolution:
    """
    A solution to a MIS problem.

    Attributes:
        instance (MISInstance): The MIS instance to which this class represents a solution.
        size (int): The number of nodes in this solution.
        node_indices (list[int]): The indices of the nodes of `instance` picked in this solution.
        nodes (list[Any]): The nodes of `instance` picked in this solution.
        frequency (float): How often this solution showed up in the measures, where 0. represents
            a solution that never showed up in the meaures and 1. a solution that showed up in all
            measures.
    """

    def __init__(self, instance: MISInstance, nodes: list[int], frequency: float):
        self.size = len(nodes)
        assert len(set(nodes)) == self.size, "All the nodes in %s should be distinct" % (nodes,)
        self.instance = instance
        self.node_indices = nodes
        self.nodes = [self.instance.index_to_node_label[i] for i in nodes]
        self.frequency = frequency

        # Note: As of this writing, self.weight is still considered a work in progress, so we
        # leave it out of the documentation.
        self.weight = calculate_weight(instance.graph, nodes)

    def draw(
        self,
        node_size: int = 600,
        highlight_color: str = "darkgreen",
        font_family: str = "Century Gothic",
    ) -> None:
        """
        Draw instance graph with solution nodes highlighted.

        Parameters:

            node_size (int): Size of drawn nodes in drawn graph. (default: 600)
            highlight_color (str): Color to highlight solution nodes with. (default: "darkgreen")
            font (str): Font type
        """
        self.instance.draw(self.node_indices, node_size, highlight_color, font_family)
