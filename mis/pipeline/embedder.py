"""
Tools to prepare the geometry (register) of atoms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pulser
from pulser import Register
from qoolqit._solvers.backends import BaseBackend

from mis.shared.types import (
    MISInstance,
)
from mis.pipeline.config import SolverConfig

from .layout import Layout


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    @abstractmethod
    def embed(self, instance: MISInstance, config: SolverConfig, backend: BaseBackend) -> Register:
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

    def embed(self, instance: MISInstance, config: SolverConfig, backend: BaseBackend) -> Register:
        device = backend.device()
        assert device is not None

        # Use Layout helper to get rescaled coordinates and interaction graph
        layout = Layout.from_device(data=instance, device=device)

        # Finally, prepare register.
        return pulser.register.Register(
            qubits={f"q{node}": pos for (node, pos) in layout.coords.items()}
        )
