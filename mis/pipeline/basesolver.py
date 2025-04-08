from __future__ import annotations

from abc import ABC, abstractmethod

from mis import MISInstance
from mis.config import SolverConfig
from mis.data import MISSolution

from .executor import Executor


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
        self.executor: Executor = Executor(config=self.config)

    @abstractmethod
    def solve(self) -> list[MISSolution]:
        """
        Solve the given MISinstance.

        Returns:
            A list of solutions, ranked from best (lowest energy) to worst (highest energy).
        """
        pass
