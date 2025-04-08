from __future__ import annotations
from typing import cast

import cplex
import networkx as nx

from mis import MISInstance, MISSolution
from mis.config import SolverConfig
from mis.pipeline import (
    BaseSolver,
    Fixtures,
    Pulse,
    Register,
    get_embedder,
    get_pulse_shaper,
)


class MISSolver(BaseSolver):
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        super().__init__(instance, config)
        self._solver: BaseSolver

        if config.method.is_quantum():
            self._solver = MISSolverQuantum(instance, config)
        else:
            self._solver = MISSolverClassical(instance, config)

    def solve(self) -> list[MISSolution]:
        return self._solver.solve()


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

    def solve(self) -> list[MISSolution]:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute,
            postprocess.

        Returns:
            MISSolution: Final result after execution and postprocessing.
        """
        self.instance = self.fixtures.preprocess()

        raise NotImplementedError

        # embedding = self.embedding()
        # pulse = self.pulse(embedding)
        # execution_result = self.execute(pulse, embedding)

        # best_bitstring = max(execution_result.items(), key=lambda x: x[1])[0]
        # solution_vector = [int(bit) for bit in best_bitstring]
        # energy = self.executor.register.device.evaluate_solution(solution_vector)

        # solution = [
        #     MISSolution(
        #         original=self.instance.graph,
        #         energy=energy,
        #         counts=execution_result,
        #         nodes=
        #     )
        # ]

        # self._solution = self.fixtures.postprocess(solution)
        # return self._solution

    # def execute(self, pulse: Pulse, embedding: Register) -> Any:
    #     """
    #     Execute the pulse schedule on the backend and retrieve the solution.

    #     Args:
    #         pulse (object): Pulse schedule or execution payload.
    #         embedding (Register): The register to be executed.

    #     Returns:
    #         Result: The solution from execution.
    #     """
    #     return asyncio.run(self.executor.submit_job(pulse, embedding))


class MISSolverClassical(BaseSolver):
    """
    Classical solver for MISproblems using brute-force search (for small instances).

    Intended for benchmarking or fallback when quantum execution is disabled.
    """

    def __init__(self, instance: MISInstance, config: SolverConfig):
        super().__init__(instance, config)

    def solve(self) -> list[MISSolution]:
        """
        Solve the MIS problem and return a single optimal solution.
        """
        graph = self.instance.graph
        if not graph.nodes:
            return []

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
        return [
            MISSolution(
                original=graph, nodes=[conversion_table[node] for node in selected_nodes], energy=0
            )
        ]
