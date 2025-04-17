from __future__ import annotations
from math import atan
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


class MISSolver(BaseSolver):
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        super().__init__(instance, config)
        self._solver: BaseSolver
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

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Solve the MIS problem and return a single optimal solution.
        """
        graph = self.instance.graph
        if not graph.nodes:
            return Execution.success([])

        mis = nx.maximal_independent_set(G=graph)
        assert isinstance(mis, list)

        # Convert back to original node labels
        return Execution.success(
            [
                MISSolution(
                    original=graph,
                    nodes=[node for node in mis],
                    energy=0,
                )
            ]
        )


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

        # FIXME: Normalize embedder.

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the MISvariables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        embedder = self.config.embedder
        assert embedder is not None
        self._register = embedder.embed(
            instance=self.instance,
            config=self.config,
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
        ranked = sorted(data.items(), key=lambda item: item[1], reverse=True)
        solutions = [
            self.fixtures.postprocess(
                MISSolution(
                    original=self.instance.graph,
                    energy=1 - atan(count),
                    # FIXME Probably not the best definition of energy
                    nodes=self._bitstring_to_nodes(bitstr),
                )
            )
            for [bitstr, count] in ranked
        ]
        return solutions

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute,
            postprocess.

        Returns:
            MISSolution: Final result after execution and postprocessing.
        """
        self.instance = self.fixtures.preprocess()

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
