from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import networkx 
import matplotlib.pyplot as plt

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
        """MIS Instance that wraps the networkx.Graph

        Args:
            graph (networkx.Graph): _description_
        """
        # FIXME: Make it work with pytorch geometric
        self.graph = graph
        self.pos = None
        
    def draw(self, nodes: list[int] | None = None, save_fig:str|bool = False):
        """Draw method provides the way to plot the Graph nodes and edges and store the figure,
        the provided input nodes will be displayed in different color

        Args:
            nodes (list[int] | None, optional): Nodes list to display in different color . Defaults to None.
            save_fig (str | bool, optional): Option to store the figure on the diskspace. Defaults to False.
        """
        if nodes is not None: 
            color_map = ['blue' if node in nodes else 'red' for node in self.graph.nodes ]
        elif nodes is None or len(node)==0:
            color_map = ['blue' for i in range(len(self.graph.nodes))]
        
        if self.pos is None:
            self.pos = networkx.spring_layout(self.graph)
        
        networkx.draw(self.graph, pos=self.pos,
                      node_size=200,
                      with_labels=True,
                      node_color=color_map,
                      nodelist=self.graph.nodes)
       

        ax = plt.gca()
        ax.margins(0.11)
        plt.tight_layout()
        plt.axis("off")
        if isinstance(save_fig, str):
            plt.savefig(f"{save_fig}.png")
        elif save_fig:
            plt.savefig("graph.png")
        plt.show()
        
        



@dataclass
class MISSolution:
    original: networkx.Graph
    nodes: list[int]
    frequency: float

    def __post_init__(self) -> None:
        # Consistency check: nodes from the list must be distinct.
        assert len(self.nodes) == len(set(self.nodes)), "All the nodes in %s should be distinct" % (
            self.nodes,
        )
        
    def draw(self, save_fig:str|bool = False):
        return MISInstance(self.original).draw(nodes=self.nodes, save_fig=save_fig)
        
