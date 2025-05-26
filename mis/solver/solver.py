from __future__ import annotations
from typing import Counter

import networkx as nx

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.execution import Execution
from mis.pipeline.fixtures import Fixtures
from mis.pipeline.embedder import DefaultEmbedder
from mis.pipeline.pulse import BasePulseShaper, DefaultPulseShaper
from mis.pipeline.targets import Pulse, Register
from mis.pipeline.config import SolverConfig


class MISSolver:
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig | None = None):
        if config is None:
            config = SolverConfig()
        self._solver: BaseSolver
        self.instance = instance
        self.config = config
        if config.backend is None:
            self._solver = MISSolverClassical(instance, config)
        else:
            self._solver = MISSolverQuantum(instance, config)

    def solve(self) -> Execution[list[MISSolution]]:
        if len(self.instance.graph.nodes) == 0:
            return Execution.success([])
        return self._solver.solve()


class MISSolverClassical(BaseSolver):
    """
    Classical (i.e. non-quantum) solver for Maximum Independent Set using
    brute-force search.

    This solver is intended for benchmarking or as a fallback when quantum
    execution is disabled.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        super().__init__(instance, config)
        self.fixtures = Fixtures(instance, self.config)

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Solve the MIS problem and return a single optimal solution.
        """

        if not self.instance.graph.nodes:
            return Execution.success([])

        preprocessed_instance = self.fixtures.preprocess()
        if len(preprocessed_instance.graph) == 0:
            # Edge case: nx.maximal_independent_set doesn't work with an empty graph.
            partial_solution = MISSolution(
                original=preprocessed_instance.graph, frequency=1.0, nodes=[]
            )
        else:
            mis = nx.approximation.maximum_independent_set(G=preprocessed_instance.graph)
            assert isinstance(mis, set)
            partial_solution = MISSolution(
                original=preprocessed_instance.graph,
                frequency=1.0,
                nodes=list(mis),
            )

        solutions = self.fixtures.postprocess([partial_solution])
        solutions = [self.fixtures.rebuild(sol) for sol in solutions]
        solutions.sort(key=lambda sol: sol.frequency, reverse=True)

        return Execution.success(solutions[: self.config.max_number_of_solutions])


class MISSolverQuantum(BaseSolver):
    """
    Quantum solver that orchestrates the solving of a MISproblem using
    embedding, pulse shaping, and quantum execution pipelines.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        """
        Initialize the MISSolver with the given problem and configuration.

        Args:
            instance (MISInstance): The MISproblem to solve.
            config (SolverConfig): Solver settings including backend and
                device.
        """
        assert config.backend is not None
        if config.embedder is None:  # FIXME: That's a side-effect on config
            config.embedder = DefaultEmbedder()
        if config.pulse_shaper is None:
            config.pulse_shaper = DefaultPulseShaper()
        if config.device is None:
            config.device = config.backend.device()

        super().__init__(instance, config)

        self.fixtures = Fixtures(instance, self.config)
        self._register: Register | None = None
        self._pulse: Pulse | None = None
        self._solution: MISSolution | None = None
        self._preprocessed_instance: MISInstance | None = None

        # FIXME: Normalize embedder.

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the MISvariables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        config: SolverConfig = self.config
        embedder = config.embedder
        assert embedder is not None
        if self._preprocessed_instance is not None:
            instance = self._preprocessed_instance
        else:
            instance = self.instance
        self._register = embedder.embed(
            instance=instance,
            config=config,
        )
        return self._register

    def pulse(self, embedding: Register) -> Pulse:
        """
        Generate the pulse sequence based on the given embedding.

        Args:
            embedding (Register): The embedded register layout.

        Returns:
            Pulse: Pulse schedule for quantum execution.
        """
        # FIXME: mypy seems to have an issue here, need to investigate.
        config: SolverConfig = self.config
        shaper = config.pulse_shaper
        assert shaper is not None
        assert isinstance(shaper, BasePulseShaper)
        self._pulse = shaper.generate(config=self.config, register=embedding)
        return self._pulse

    def _bitstring_to_nodes(self, bitstring: str) -> list[int]:
        result: list[int] = []
        for i, c in enumerate(bitstring):
            if c == "1":
                result.append(i)
        return result

    def _process(self, data: Counter[str]) -> list[MISSolution]:
        """
        Process bitstrings into solutions.
        """
        assert self._preprocessed_instance is not None
        total = data.total()
        if len(data) == 0:
            # No data? This can only happen if the graph was empty in the first place.
            # In turn, this can happen if preprocessing was really lucky and managed
            # to whittle down the original graph to an empty graph. But we need at least one
            # partial solution to be able to rebuild an MIS, so we handle this edge
            # case by injecting an empty solution.
            postprocessed = [
                MISSolution(original=self._preprocessed_instance.graph, frequency=1, nodes=[])
            ]

            # No noise here, since there wasn't any quantum measurement, so no
            # postprocessing.
        else:
            raw = [
                MISSolution(
                    original=self._preprocessed_instance.graph,
                    frequency=count
                    / total,  # Note: If total == 0, the list is empty, so this line is never called.
                    nodes=self._bitstring_to_nodes(bitstr),
                )
                for [bitstr, count] in data.items()
            ]

            # Postprocess to get rid of quantum noise.
            postprocessed = self.fixtures.postprocess(raw)

        # Then rebuild any partial solution into solutions on the full graph.
        rebuilt = [self.fixtures.rebuild(r) for r in postprocessed]

        # And present the most interesting solutions first.
        rebuilt.sort(key=lambda sol: sol.frequency, reverse=True)
        return rebuilt[: self.config.max_number_of_solutions]

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute,
            postprocess.

        Returns:
            MISSolution: Final result after execution and postprocessing.
        """
        self._preprocessed_instance = self.fixtures.preprocess()
        if len(self._preprocessed_instance.graph) == 0:
            # Edge case: we cannot process an empty register.
            return Execution.success(self._process(Counter()))
        embedding = self.embedding()
        pulse = self.pulse(embedding)
        execution_result = self.execute(pulse, embedding)
        return execution_result.map(self._process)

    def execute(self, pulse: Pulse, embedding: Register) -> Execution[Counter[str]]:
        """
        Execute the pulse schedule on the backend and retrieve the solution.

        Args:
            pulse (object): Pulse schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            Result: The solution from execution.
        """
        return self.executor.submit_job(pulse, embedding)
