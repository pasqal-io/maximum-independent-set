from __future__ import annotations
from typing import Counter, Callable
import networkx as nx
import copy

from pulser import Pulse, Register
from qoolqit._solvers.backends import QuantumProgram, get_backend, BackendConfig, BaseBackend
from mis.shared.types import MISInstance, MISSolution, MethodType
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.fixtures import Fixtures
from mis.pipeline.embedder import DefaultEmbedder
from mis.pipeline.pulse import DefaultPulseShaper
from mis.pipeline.config import SolverConfig
from mis.solver.greedymapping import GreedyMapping
from mis.pipeline.layout import Layout
from mis.shared.graphs import calculate_weight, remove_neighborhood


def _extract_backend(config: SolverConfig) -> BaseBackend:
    if config.backend is None:
        raise ValueError("Invalid config.backend: expecting a backend to run in quantum mode")
    elif isinstance(config.backend, BaseBackend):
        return config.backend
    elif isinstance(config.backend, BackendConfig):
        return get_backend(config.backend)
    else:
        raise ValueError("Invalid config.backend")


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
            solver_factory: type[BaseSolver] = MISSolverClassical
        else:
            solver_factory = MISSolverQuantum
        if config.method == MethodType.GREEDY:
            self._solver = GreedyMISSolver(instance, config, solver_factory)
        else:
            self._solver = solver_factory(instance, config)

    def solve(self) -> list[MISSolution]:
        if len(self.instance.graph.nodes) == 0:
            return []
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
        if config.backend is not None:
            raise ValueError(
                "MISSolverClassical may not be used in non-quantum mode (e.g. if backend=None)"
            )
        self.fixtures = Fixtures(instance, self.config)

    def solve(self) -> list[MISSolution]:
        """
        Solve the MIS problem and return a single optimal solution.
        """

        if not self.instance.graph.nodes:
            return []

        preprocessed_instance = self.fixtures.preprocess()
        if len(preprocessed_instance.graph) == 0:
            # Edge case: nx.maximal_independent_set doesn't work with an empty graph.
            partial_solution = MISSolution(instance=preprocessed_instance, frequency=1.0, nodes=[])
        else:
            mis = nx.approximation.maximum_independent_set(G=preprocessed_instance.graph)
            assert isinstance(mis, set)
            partial_solution = MISSolution(
                instance=preprocessed_instance,
                frequency=1.0,
                nodes=list(mis),
            )

        solutions = self.fixtures.postprocess([partial_solution])
        solutions = [self.fixtures.rebuild(sol) for sol in solutions]
        solutions.sort(key=lambda sol: sol.frequency, reverse=True)

        return solutions[: self.config.max_number_of_solutions]


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
        self.backend = _extract_backend(config)
        self._register: Register | None = None
        self._pulse: Pulse | None = None
        self._solution: MISSolution | None = None
        self._preprocessed_instance: MISInstance | None = None
        self._embedder = config.embedder if config.embedder is not None else DefaultEmbedder()
        self._shaper = (
            config.pulse_shaper if config.pulse_shaper is not None else DefaultPulseShaper()
        )

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the MISvariables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        config: SolverConfig = self.config
        if self._preprocessed_instance is not None:
            instance = self._preprocessed_instance
        else:
            instance = self.instance
        self._register = self._embedder.embed(
            instance=instance,
            config=config,
            backend=self.backend,
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
        self._pulse = self._shaper.generate(
            config=self.config, register=embedding, backend=self.backend, instance=self.instance
        )
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
                MISSolution(instance=self._preprocessed_instance, frequency=1, nodes=[])
            ]

            # No noise here, since there wasn't any quantum measurement, so no
            # postprocessing.
        else:
            raw = [
                MISSolution(
                    instance=self._preprocessed_instance,
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

    def solve(self) -> list[MISSolution]:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute,
            postprocess.

        Returns:
            MISSolution: Final result after execution and postprocessing.
        """
        self._preprocessed_instance = self.fixtures.preprocess()
        if len(self._preprocessed_instance.graph) == 0:
            # Edge case: we cannot process an empty register.
            return self._process(Counter())
        embedding = self.embedding()
        pulse = self.pulse(embedding)
        execution_result = self.execute(pulse, embedding)
        return self._process(execution_result)

    def execute(self, pulse: Pulse, register: Register) -> Counter[str]:
        """
        Execute the pulse schedule on the backend and retrieve the solution.

        Args:
            pulse (object): Pulse schedule or execution payload.
            embedding (Register): The register to be executed.

        Returns:
            Result: The solution from execution.
        """
        program = QuantumProgram(register=register, pulse=pulse, device=self.backend.device())
        counts = self.backend.run(program=program, runs=self.config.runs).counts
        assert isinstance(counts, Counter)  # Not sure why mypy expects that `counts` is `Any`.
        return counts


class GreedyMISSolver(BaseSolver):
    """
    A recursive solver that maps an MISInstance onto a physical layout using greedy subgraph embedding.
    Uses an internal exact solver for small subproblems and a greedy decomposition strategy for larger graphs.

    Note:
        This solver uses recursive decomposition via `_solve_recursive()`.
        Python's default recursion limit is 1000 (see `sys.getrecursionlimit()`).
        For large graphs or deep recursion trees, this limit may be hit.
    """

    def __init__(
        self,
        instance: MISInstance,
        config: SolverConfig,
        solver_factory: Callable[[MISInstance, SolverConfig], BaseSolver],
    ) -> None:
        """
        Initializes the GreedyMISSolver with a given MIS problem instance and a base solver.

        Args:
            instance (MISInstance): The full MIS problem instance to solve.
            config (SolverConfig): Solver settings including backend and
                device.
            solver_factory (Callable[[MISInstance, SolverConfig], BaseSolver]):
                The solver factory (used for solving subproblems recursively).
        """
        super().__init__(instance, config)
        if config.backend is None:
            # Classical mode
            self.backend = None
        else:
            # Quantum mode
            self.backend = _extract_backend(config)
        self.solver_factory = solver_factory
        self.layout = self._build_layout()

    def _build_layout(self) -> Layout:
        """
        Constructs the Layout object based on config:

        - If this GreedyMISSolver is configured for quantum, uses the device information.
        - Otherwise, use a default distance 1.0 for layout generation.

        Returns:
            Layout: The constructed layout.
        """
        if self.backend is None:
            # A default layout for the classical solver.
            return Layout(data=self.instance, rydberg_blockade=1.0)
        return Layout.from_device(data=self.instance, device=self.backend.device())

    def solve(self) -> list[MISSolution]:
        """
                Entry point for solving the full MISInstance using recursive greedy decomposition.
                Greedy MIS Solver (recursive MIS via subgraph decomposition)

                Algorithm:
                Input: MISInstance (graph), SolverConfig (with greedy & quantum options),
                    SolverFactory (used for exact or quantum solving)
                Output: best_solution: approx MIS set

                1. If graph size ≤ exact_solving_threshold:
                    - Solve directly using base solver
                    - Return result
                2. Generate greedy mappings (subgraph_quantity of them)
                3. For each mapping:
                    a. Build layout subgraph
                    b. Solve subgraph using solver_factory
                    c. Map solution back to original graph nodes
                    d. For each MIS solution:
                        i. Remove closed neighborhood from graph
                        ii. Solve recursively on the remainder
                        iii. Combine with current MIS
                        iv. If better than best_solution → update

                4. Return best_solution (or empty solution if none found)
        `
                Returns:
                    Execution containing a list of optimal or near-optimal MIS solutions.
        """
        return self._solve_recursive(self.instance)

    def _solve_recursive(self, instance: MISInstance) -> list[MISSolution]:
        """
        Recursively solves an MISInstance:
        - Uses exact backtracking for small subgraphs.
        - Otherwise partitions and solves using greedy mapping and recursion.

        Args:
            instance: The current MISInstance to solve.

        Returns:
            Execution containing a list of solutions.
        """
        graph = instance.graph
        if len(graph) <= self.config.greedy.default_solving_threshold:  # type: ignore[union-attr]
            solver = self.solver_factory(instance, self.config)
            return solver.solve()

        # these mappings from from graph nodes - to - layout nodes
        mappings = self._generate_subgraphs(graph)
        best_solution: MISSolution | None = None

        for mapping in mappings:
            layout_subgraph = self._generate_layout_graph(graph, mapping)
            sub_instance = MISInstance(graph=layout_subgraph)
            solver = self.solver_factory(sub_instance, self.config)

            # Note:
            # this forces the computation to be in sync.
            # TODO: Align with async SDK - when it is available in the future.
            # results in a solution in forms of layout nodes
            results = solver.solve()

            # inverse mappings from from layout nodes - to - graph nodes
            inv_map = {v: k for k, v in mapping.items()}

            # collections of graph nodes for each solution.
            # based on the collected layout nodes from the solver.solve()
            current_mis_bag = [
                [inv_map[value] for value in mis_lattice.nodes] for mis_lattice in results
            ]

            for current_mis in current_mis_bag:
                reduced_graph = remove_neighborhood(graph, current_mis)
                remainder_instance = MISInstance(reduced_graph)
                remainder_solutions = self._solve_recursive(remainder_instance)

                for rem_sol in remainder_solutions:
                    combined_nodes = current_mis + rem_sol.nodes
                    if (best_solution is None) or (
                        calculate_weight(self.instance.graph, combined_nodes) > best_solution.weight
                    ):
                        best_solution = MISSolution(
                            instance=instance, nodes=combined_nodes, frequency=1.0
                        )

        if best_solution is None:
            return [MISSolution(instance=instance, nodes=[], frequency=0)]
        return [best_solution]

    def _generate_subgraphs(self, graph: nx.Graph) -> list[dict[int, int]]:
        """
        Generates subgraph mappings using greedy layout placement.
        The largest mappings are returned because they represent the most successful embedding
        attempt of the original graph onto the quantum-executable lattice, for higher-quality
        approximations of the original MIS problem.

        Args:
            graph: The input logical graph.

        Returns:
            List of mappings from logical node → layout node.
        """
        mappings = []
        for node in graph.nodes():
            mapper = GreedyMapping(
                instance=MISInstance(graph),
                layout=copy.deepcopy(self.layout),
                previous_subgraphs=[],
            )
            mapping = mapper.generate(starting_node=node)
            mappings.append(mapping)
        return sorted(mappings, key=lambda m: len(m), reverse=True)[
            : self.config.greedy.subgraph_quantity  # type: ignore[union-attr]
        ]

    def _generate_layout_graph(self, graph: nx.Graph, mapping: dict[int, int]) -> nx.Graph:
        """
        Creates a subgraph in layout space from a logical-to-layout mapping.

        Args:
            graph: The logical graph.
            mapping: Mapping from logical nodes to layout node indices.

        Returns:
            A new NetworkX graph in physical layout space.
        """
        G = nx.Graph()
        for logical, physical in mapping.items():
            weight = graph.nodes[logical].get("weight", 0.0)
            pos = self.layout.graph.nodes[physical].get("pos", (0, 0))
            G.add_node(physical, weight=weight, pos=pos)

        for _, physical in mapping.items():
            for neighbor in self.layout.graph.neighbors(physical):
                if neighbor in mapping.values():
                    G.add_edge(physical, neighbor)

        return G
