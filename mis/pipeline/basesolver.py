"""
Shared definitions for solvers.

This module is useful mostly for users interested in writing
new solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from mis.pipeline.config import SolverConfig
from mis.shared.types import MISInstance, MISSolution


class BaseSolver(ABC):
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, pulse shaping,
    and execution of MISproblems.

    The BaseSolver also provides a method to execute the Pulse and
    Register
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        """
        Initialize the solver with the MISinstance and configuration.

        Args:
            instance (MISInstance): The MISproblem to solve.
            config (SolverConfig): Configuration settings for the solver.
        """
        self.instance: MISInstance = instance
        self.config: SolverConfig = config

    @abstractmethod
    def solve(self) -> list[MISSolution]:
        """
        Solve the given MISinstance.

        Returns:
            A list of solutions, ranked from best (lowest energy) to worst
            (highest energy).
        """
        pass
