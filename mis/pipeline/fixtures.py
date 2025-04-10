from __future__ import annotations

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.config import SolverConfig


class Fixtures:
    """
    Handles all preprocessing and postprocessing logic for MIS problems.

    This class allows centralized transformation or validation of the problem
    instance before solving, and modification or annotation of the solution after solving.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        """
        Initialize the fixture handler with the MIS instance and solver config.

        Args:
            instance (MISInstance): The problem instance to process.
            config (SolverConfig): Solver configuration, which may include flags
                                   for enabling or customizing processing behavior.
        """
        self.instance = instance
        self.config = config

    def preprocess(self) -> MISInstance:
        """
        Apply preprocessing steps to the MIS instance before solving.

        Returns:
            MISInstance: The processed or annotated instance.
        """
        return self.instance

    def postprocess(self, solution: MISSolution) -> MISSolution:
        """
        Apply postprocessing steps to the MIS solution after solving.

        Args:
            solution (MISSolution): The raw solution from a solver.

        Returns:
            MISSolution: The cleaned or transformed solution.
        """
        return solution
