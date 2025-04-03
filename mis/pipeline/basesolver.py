from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from mis import QUBOInstance
from mis.config import SolverConfig
from mis.data import QUBOSolution

from .executor import Executor
from .targets import Pulse, Register


class BaseSolver(ABC):
    """
    Abstract base class for all solvers (quantum or classical).

    Provides the interface for solving, embedding, pulse shaping,
    and execution of MISproblems.

    The BaseSolver also provides a method to execute the Pulse and
    Register
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        """
        Initialize the solver with the MISinstance and configuration.

        Args:
            instance (QUBOInstance): The MISproblem to solve.
            config (SolverConfig): Configuration settings for the solver.
        """
        self.instance: QUBOInstance = instance
        self.config: SolverConfig = config
        self.executor: Executor = Executor(config=self.config)

    @abstractmethod
    def solve(self) -> QUBOSolution:
        """
        Solve the given MISinstance.

        Returns:
            QUBOSolution: The result of the optimization.
        """
        pass

    @abstractmethod
    def embedding(self) -> Register:
        """
        Generate or retrieve an embedding for the MISinstance.

        Returns:
            dict: Embedding information for the instance.
        """
        pass

    @abstractmethod
    def pulse(self, embedding: Register) -> Pulse:
        """
        Generate a pulse schedule for the quantum device based on the embedding.

        Args:
            embedding (dict): Embedding information.

        Returns:
            object: Pulse schedule or related data.
        """
        pass

    def execute(self, pulse: Pulse, embedding: Register) -> Any:
        """
        Execute the pulse schedule on the backend and retrieve the solution.

        Args:
            pulse (object): Pulse schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            Result: The solution from execution.
        """
        return asyncio.run(self.executor.submit_job(pulse, embedding))
