from __future__ import annotations

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.config import SolverConfig
from mis.pipeline.preprocessor import BasePreprocessor
from mis.pipeline.postprocessor import BasePostprocessor


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
        self.postprocessor: BasePostprocessor | None = None
        if self.config.postprocessor is not None:
            self.postprocessor = self.config.postprocessor()

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

    def rebuild(self, solution: MISSolution) -> MISSolution:
        """
        Apply any pending rebuild operations to convert solutions
        on preprocessed graphs into solutions on the original graph.

        Args:
            solution (MISSolution): The raw solution from a solver.

        Returns:
            MISSolution: The cleaned or transformed solution.
        """
        if self.preprocessor is None:
            return solution
        # If we have preprocessed the graph, we end up with a solution
        # that only works for the preprocessed graph.
        #
        # At this stage, we need to call the preprocessor's rebuilder to
        # expand this to a solution on the original graph.
        nodes = self.preprocessor.rebuild(set(solution.nodes))
        return MISSolution(instance=self.instance, nodes=list(nodes), frequency=solution.frequency)

    def postprocess(self, solutions: list[MISSolution]) -> list[MISSolution]:
        if self.postprocessor is None:
            return solutions

        # Run postprocessing.
        postprocessed_solutions: dict[str, MISSolution] = {}
        for solution in solutions:
            processed_solution = self.postprocessor.postprocess(solution)
            if processed_solution is None:
                continue
            key = f"{processed_solution.nodes}"  # This is a bit of a waste, we could have used bistrings.
            previous = postprocessed_solutions.get(key)
            if previous is None:
                postprocessed_solutions[key] = processed_solution
            else:
                # Merge the two solutions.
                previous.frequency += processed_solution.frequency

        return list(postprocessed_solutions.values())
