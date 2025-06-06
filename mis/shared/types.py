from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
import networkx
import matplotlib.pyplot as plt
from mis.shared.graphs import calculate_weight


class BackendType(str, Enum):
    """
    Type of backend to use for solving the MIS
    """

    QUTIP = "qutip"
    REMOTE_QPU = "remote_qpu"
    REMOTE_EMUMPS = "remote_emumps"


class MethodType(str, Enum):
    EAGER = "eager"
    GREEDY = "greedy"


class MISInstance:
    def __init__(self, graph: networkx.Graph):
        # FIXME: Make it work with pytorch geometric
        self.graph = graph

    def draw(
        self, nodes: list[int] | None = None, node_size: int = 600, highlight_color: str = "red"
    ) -> None:
        """
        Draw instance graph with highlighted nodes.

        Parameters:

            nodes (list[int]): List of nodes to highlight.
            node_size (int): Size of drawn nodes in drawn graph. (default: 600)
            highlight_color (str): Color to highlight nodes with. (default: "red")
        """
        # Obtain a view of all nodes
        all_nodes = self.graph.nodes
        # Compute graph layout
        node_positions = networkx.kamada_kawai_layout(self.graph)
        # Keyword dictionaries to customize appearance
        highlighted_node_kwds = {"node_color": highlight_color, "node_size": node_size}
        unhighlighted_node_kwds = {
            "node_color": "white",
            "edgecolors": "black",
            "node_size": node_size,
        }
        if nodes:  # If nodes is not empty
            nodeset = set(nodes)  # Create a set from node list for easier operations
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
                self.graph, node_positions, nodelist=nodes, **highlighted_node_kwds
            )
            # Draw unhighlighted nodes
            networkx.draw_networkx_nodes(
                self.graph,
                node_positions,
                nodelist=list(nodes_complement),
                **unhighlighted_node_kwds,
            )
        else:
            networkx.draw_networkx_nodes(
                self.graph, node_positions, nodelist=list(all_nodes), **unhighlighted_node_kwds
            )
        # Draw node labels
        networkx.draw_networkx_labels(self.graph, node_positions)
        # Draw edges
        networkx.draw_networkx_edges(self.graph, node_positions)
        plt.tight_layout()
        plt.axis("off")
        plt.show()


@dataclass
class MISSolution:
    original: networkx.Graph
    nodes: list[int]
    frequency: float
    weight: float = field(init=False)

    def __post_init__(self) -> None:
        # Consistency check: nodes from the list must be distinct.
        assert len(self.nodes) == len(set(self.nodes)), "All the nodes in %s should be distinct" % (
            self.nodes,
        )
        self.weight = calculate_weight(self.original, self.nodes)

    def draw(self, node_size: int = 600, highlight_color: str = "red") -> None:
        """
        Draw instance graph with solution nodes highlighted.

        Parameters:

            node_size (int): Size of drawn nodes in drawn graph. (default: 600)
            highlight_color (str): Color to highlight nodes with. (default: "red")
        """
        MISInstance(self.original).draw(self.nodes, node_size, highlight_color)
