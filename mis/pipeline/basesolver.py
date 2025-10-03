"""
Shared definitions for solvers.

This module is useful mostly for users interested in writing
new solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from mis.pipeline.config import SolverConfig
from mis.shared.types import MISInstance, MISSolution
from qoolqit._solvers import QuantumProgram


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

    def quantum_program(self) -> QuantumProgram:
        """
        If this solver executes a quantum program, return it.

        This method is meant mostly for pedagogical or debugging purposes.
        """
        raise NotImplementedError
