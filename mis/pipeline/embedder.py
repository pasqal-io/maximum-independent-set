from __future__ import annotations

from abc import ABC, abstractmethod

from mis import QUBOInstance
from mis.config import SolverConfig

from .targets import Register


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        """
        Args:
            instance (QUBOInstance): The MISproblem to embed.
            config (SolverConfig): The Solver Configuration.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.register: Register | None = None

    @abstractmethod
    def embed(self) -> Register:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        pass


class FirstEmdedder(BaseEmbedder):
    """
    A simple embedder
    """

    def embed(self) -> Register:
        raise NotImplementedError


def get_embedder(instance: QUBOInstance, config: SolverConfig) -> BaseEmbedder:
    """
    Method that returns the correct embedder based on configuration.
    The correct embedding method can be identified using the config, and an
    object of this embedding can be returned using this function.

    Args:
        instance (QUBOInstance): The MISproblem to embed.
        config (Device): The quantum device to target.

    Returns:
        (BaseEmbedder): The representative embedder object.
    """

    return FirstEmdedder(instance, config)
