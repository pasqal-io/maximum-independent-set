from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import networkx


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


@dataclass
class MISSolution:
    original: networkx.Graph
    nodes: list[int]
    frequency: float
