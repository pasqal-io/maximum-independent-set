from __future__ import annotations

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.config import SolverConfig
from mis.pipeline.preprocessor import BasePreprocessor


class Fixtures:
    """
    Handles all preprocessing and postprocessing logic for MIS problems.

    This class allows centralized transformation or validation of the problem
    instance before solving, and modification or annotation of the solution
    after solving.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        """
        Initialize the fixture handler with the MIS instance and solver config.

        Args:
            instance: The problem instance to process.
            config: Solver configuration, which may include
                flags for enabling or customizing processing behavior.
        """
        self.instance = instance
        self.config = config
        self.preprocessor: BasePreprocessor | None = None
        if self.config.preprocessor is not None:
            self.preprocessor = self.config.preprocessor(instance.graph)

    def preprocess(self) -> MISInstance:
        """
        Apply preprocessing steps to the MIS instance before solving.

        Returns:
            MISInstance: The processed or annotated instance.
        """
        if self.preprocessor is not None:
            graph = self.preprocessor.preprocess()
            return MISInstance(graph)
        return self.instance

    def postprocess(self, solution: MISSolution) -> MISSolution:
        """
        Apply postprocessing steps to the MIS solution after solving.

        Args:
            solution (MISSolution): The raw solution from a solver.

        Returns:
            MISSolution: The cleaned or transformed solution.
        """
        if self.preprocessor is not None:
            # If we have preprocessed the graph, we end up with a solution
            # that only works for the preprocessed graph.
            #
            # At this stage, we need to call the preprocessor's rebuilder to
            # expand this to a solution on the original graph.
            nodes = self.preprocessor.rebuild(set(solution.nodes))
            return MISSolution(
                original=self.instance.graph, nodes=list(nodes), energy=solution.energy
            )
        return solution
