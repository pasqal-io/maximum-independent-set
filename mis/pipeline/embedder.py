"""
Tools to prepare the geometry (register) of atoms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qoolqit import Register

from mis.shared.types import (
    MISInstance,
)
from mis.pipeline.config import SolverConfig

from .layout import Layout


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Qoolqit devices.
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

        # Use Layout helper to get rescaled coordinates and interaction graph
        layout = Layout.from_device(data=instance, device=device)

        # Finally, prepare register.
        conversion_factor = device.converter.factors[2]
        return Register(
            qubits={f"q{node}": pos / conversion_factor for (node, pos) in layout.coords.items()}
        )
