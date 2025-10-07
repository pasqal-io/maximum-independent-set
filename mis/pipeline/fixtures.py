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
            self.preprocessor = self.config.preprocessor(config, instance.graph)
        self.postprocessor: BasePostprocessor | None = None
        if self.config.postprocessor is not None:
            self.postprocessor = self.config.postprocessor(config)

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

    def _solution_key(self, solution: MISSolution | list[int] | frozenset[int]) -> str:
        """
        Generate a signature key to deduplicate solutions.
        """
        if isinstance(solution, frozenset):
            indices = list(solution)
        elif isinstance(solution, MISSolution):
            indices = solution.node_indices
        else:
            indices = solution
        return f"{sorted(indices)}"

    def postprocess(self, solutions: list[MISSolution]) -> list[MISSolution]:
        """
        Apply postprocessing steps to the MIS instance after solving.

        These postprocessing steps consist in:
        - first, any explicit postprocessor we have setup in `config`, used e.g.
            to compensate for noise
        - second, any implicit postprocessors (aka rebuilders) previously registered
            by the preprocessor to convert solutions on preprocessed graphs
            into solutions on the original graph, executed in the reverse order
            in which they were registered.

        Note: Neither of the operations maintains the frequency. Past this point,
            the frequency of a solution may end up being > 1.0.

        Args:
            solutions (MISSolution): The solutions returned by the quantum device.

        Returns:
            list[MISSolution]: The cleaned or transformed solution.
        """

        # Start with postprocessing, to compensate for noise or rounding errors.
        if self.postprocessor is None:
            bag: dict[str, MISSolution] = {
                self._solution_key(solution): solution for solution in solutions
            }
        else:
            bag = {}
            for solution in solutions:
                processed = self.postprocessor.postprocess(solution)
                if processed is None:
                    # Disregard solution.
                    continue
                key = self._solution_key(processed)
                existing = bag.get(key)
                if existing is None:
                    bag[key] = processed
                else:
                    # Deduplicate solution.
                    existing.frequency += processed.frequency

        # Now, any remaining solution should be a valid (w)MIS. However,
        # these solutions may be valid only for preprocessed graphs.
        # Rebuild these into solutions on the original graph.
        old_bag = bag
        if self.preprocessor is None:
            bag = old_bag
        else:
            bag = {}
            for partial in old_bag.values():
                for nodes in self.preprocessor.rebuild(frozenset(partial.nodes)):
                    key = self._solution_key(nodes)
                    existing = bag.get(key)
                    if existing is None:
                        bag[key] = MISSolution(
                            instance=self.instance, nodes=list(nodes), frequency=partial.frequency
                        )
                    else:
                        # Deduplicate solution.
                        existing.frequency += partial.frequency

        return list(bag.values())
