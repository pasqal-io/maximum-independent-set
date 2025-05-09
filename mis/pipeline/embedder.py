from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
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
        device = config.device
        assert device is not None

        # Layout based on edges.
        positions = nx.spring_layout(instance.graph)

        # Rescale to ensure that minimal distances are respected.
        distances = [
            np.linalg.norm(positions[v1] - positions[v2])
            for v1 in instance.graph.nodes()
            for v2 in instance.graph.nodes()
            if v1 != v2
        ]
        if len(distances) != 0:
            min_distance = np.min(distances)
            if min_distance < device.min_atom_distance:
                multiplier = device.min_atom_distance / min_distance
                positions = {i: v * multiplier for (i, v) in positions.items()}

        # Finally, prepare register.
        reg = pulser.register.Register(
            qubits={f"q{node}": pos for (node, pos) in positions.items()}
        )
        return Register(device=device, register=reg, graph=instance.graph)
