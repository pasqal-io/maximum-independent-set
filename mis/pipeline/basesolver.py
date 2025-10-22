"""
Shared definitions for solvers.

This module is useful mostly for users interested in writing
new solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from mis.pipeline.config import SolverConfig
from mis.shared.types import MISInstance, MISSolution
from qoolqit import Register, Drive


class BaseSolver(ABC):
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, drive shaping,
    and execution of MISproblems.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        """
        Initialize the solver with the MISinstance and configuration.

        Args:
            instance (MISInstance): The MISproblem to solve.
            config (SolverConfig): Configuration settings for the solver.
        """
        self.original_instance: MISInstance = instance
        self.config: SolverConfig = config

    @abstractmethod
    def solve(self) -> list[MISSolution]:
        """
        Solve the given MISinstance.

        Arguments:
            instance: if None (default), use the original instance passed during
            initialization. Otherwise, pass a custom instance. Used e.g. for
            preprocessing.

        Returns:
            A list of solutions, ranked from best (lowest energy) to worst
            (highest energy).
        """
        pass

    @abstractmethod
    def embedding(self) -> Register:
        """
        Generate or retrieve an embedding for the instance.

        Returns:
            dict: Embedding information for the instance.
        """
        pass

    @abstractmethod
    def drive(self, embedding: Register) -> Drive:
        """
        Generate a drive for the quantum device based on the embedding.

        Args:
            embedding (Register): Embedding information.

        Returns:
            Drive: drive for quantum program.
        """
        pass
