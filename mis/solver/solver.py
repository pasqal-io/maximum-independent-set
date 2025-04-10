from __future__ import annotations
from math import atan
from typing import Counter, cast

import cplex
import networkx as nx

from mis.shared.types import (
    MISInstance,
    MISSolution
)
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.embedder import get_embedder
from mis.pipeline.execution import Execution
from mis.pipeline.fixtures import Fixtures
from mis.pipeline.pulse import get_pulse_shaper
from mis.pipeline.targets import Pulse, Register
from mis.config import SolverConfig


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

        c = cplex.Cplex()

        # Relabel the graph with consecutive integers
        solved_graph = nx.convert_node_labels_to_integers(graph)

        # Setting variables as binary
        c.variables.add(
            names=[str(node) for node in solved_graph.nodes()],
            types=cast(  # Type annotation in cplex is wrong
                str, [c.variables.type.binary] * len(solved_graph.nodes())
            ),
        )

        # Independence constraints
        c.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=[str(u), str(v)], val=[1.0, 1.0])
                for u, v in solved_graph.edges()
            ],
            senses=["L"] * solved_graph.number_of_edges(),
            rhs=[1.0] * solved_graph.number_of_edges(),
        )

        # Objective function definition
        c.objective.set_linear([(str(node), 1) for node in solved_graph.nodes()])
        c.objective.set_sense(c.objective.sense.maximize)

        # Solve MIP without logs
        c.set_log_stream(None)
        c.set_results_stream(None)
        c.set_warning_stream(None)
        c.solve()

        # Extract solution
        solution_values = c.solution.get_values()
        selected_nodes = [node for node, value in enumerate(solution_values) if value >= 0.9]

        # Convert back to original node labels
        conversion_table = list(graph.nodes())
        return Execution.success([
            MISSolution(
                original=graph,
                nodes=[conversion_table[node] for node in selected_nodes],
                energy=0
            )
        ])


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
        super().__init__(instance, config)

        self.fixtures = Fixtures(instance, self.config)
        self.embedder = get_embedder(instance, self.config)
        self.pulse_shaper = get_pulse_shaper(instance, self.config)

        self._register: Register | None = None
        self._pulse: Pulse | None = None
        self._solution: MISSolution | None = None

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the MISvariables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        self._register = self.embedder.embed()
        return self._register

    def pulse(self, embedding: Register) -> Pulse:
        """
        Generate the pulse sequence based on the given embedding.

        Args:
            embedding (Register): The embedded register layout.

        Returns:
            Pulse: Pulse schedule for quantum execution.
        """
        self._pulse = self.pulse_shaper.generate(embedding)
        return self._pulse

    def _bitstring_to_nodes(self, bitstring: str) -> list[int]:
        result: list[int] = []
        for i, c in enumerate(bitstring):
            if c == '1':
                result.append(i)
        return result

    def _process(self, data: Counter[str]) -> list[MISSolution]:
        ranked = sorted(
            data.items(),
            key=lambda item: item[1],
            reverse=True)
        solutions = [self.fixtures.postprocess(MISSolution(
            original=self.instance.graph,
            energy=1-atan(count),  # FIXME Probably not the best definition of energy
            nodes=self._bitstring_to_nodes(bitstr),
        )) for [bitstr, count] in ranked]
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

    def execute(self,
                pulse: Pulse,
                embedding: Register) -> Execution[Counter[str]]:
        """
        Execute the pulse schedule on the backend and retrieve the solution.

        Args:
            pulse (object): Pulse schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            Result: The solution from execution.
        """
        return self.executor.submit_job(pulse, embedding)

