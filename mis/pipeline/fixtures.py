from __future__ import annotations

from mis import QUBOInstance, QUBOSolution
from mis.config import SolverConfig


class Fixtures:
    """
    Handles all preprocessing and postprocessing logic for MIS problems.

    This class allows centralized transformation or validation of the problem
    instance before solving, and modification or annotation of the solution after solving.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        """
        Initialize the fixture handler with the MIS instance and solver config.

        Args:
            instance (QUBOInstance): The problem instance to process.
            config (SolverConfig): Solver configuration, which may include flags
                                   for enabling or customizing processing behavior.
        """
        self.instance = instance
        self.config = config

    def preprocess(self) -> QUBOInstance:
        """
        Apply preprocessing steps to the MIS instance before solving.

        Returns:
            QUBOInstance: The processed or annotated instance.
        """
        return self.instance

    def postprocess(self, solution: QUBOSolution) -> QUBOSolution:
        """
        Apply postprocessing steps to the MIS solution after solving.

        Args:
            solution (QUBOSolution): The raw solution from a solver.

        Returns:
            QUBOSolution: The cleaned or transformed solution.
        """
        return solution
