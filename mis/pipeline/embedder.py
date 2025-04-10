from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import pulser

from mis.shared.types import (
    MISInstance,
)
from mis.pipeline.config import SolverConfig

from .targets import Register


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    @abstractmethod
    def embed(self, instance: MISInstance, config: SolverConfig) -> Register:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        pass


class DefaultEmbedder(BaseEmbedder):
    """
    A simple embedder
    """

    def embed(self, instance: MISInstance, config: SolverConfig) -> Register:
        # Layout based on edges.
        positions = nx.spring_layout(instance.graph)
        # FIXME: Rescale if necessary.
        device = config.device
        assert device is not None

        reg = pulser.register.Register(qubits={
            f"q{node}": pos for (node, pos) in positions.items()
            })
        return Register(device=device, register=reg)
