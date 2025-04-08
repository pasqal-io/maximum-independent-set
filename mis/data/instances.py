from dataclasses import dataclass
import networkx


class MISInstance:
    def __init__(self, graph: networkx.Graph):
        # FIXME: Make it work with pytorch geometric
        self.graph = graph


@dataclass
class MISSolution:
    original: networkx.Graph
    nodes: list[int]
    energy: float
