"""
Shared definitions for solvers.

This module is useful mostly for users interested in writing
new solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from mis.pipeline.config import SolverConfig
from mis.shared.types import MISInstance, MISSolution
from pulser import Register, Pulse
from mis._backends import Detuning


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

    @abstractmethod
    def embedding(self) -> Register:
        """
        Generate or retrieve an embedding for the instance.

        Returns:
            dict: Embedding information for the instance.
        """
        pass

    @abstractmethod
    def pulse(self, embedding: Register) -> Pulse:
        """
        Generate a pulse schedule for the quantum device based on the embedding.

        Args:
            embedding (Register): Embedding information.

        Returns:
            Pulse: Pulse schedule.
        """
        pass

    def detuning(self, embedding: Register) -> list[Detuning]:
        """Return detunings to be executed alongside the pulses.

        Args:
            embedding (Register): Embedding information.

        Returns:
            list[Detuning]: The list of detunings.
        """
        return []
