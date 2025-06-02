from __future__ import annotations

from typing import Callable
import networkx as nx
import copy

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.execution import Execution
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.config import SolverConfig
from mis.solver.greedymapping import GreedyMapping
from mis.pipeline.layout import Layout
from mis.shared.utils import calculate_weight



class GreedyMISSolver(BaseSolver):
    """
    A recursive solver that maps an MISInstance onto a physical layout using greedy subgraph embedding.
    Uses an internal exact solver for small subproblems and a greedy decomposition strategy for larger graphs.
    """

    def __init__(
        self,
        instance: MISInstance,
        config: SolverConfig,
        solver_class: Callable[[MISInstance, SolverConfig], BaseSolver],
    ) -> None:
        """
        Initializes the GreedyMISSolver with a given MIS problem instance and a base solver.

        Args:
            instance: The full MIS problem instance to solve.
            config (SolverConfig): Solver settings including backend and
                device.
            solver: The base solver (used for solving subproblems recursively).
        """
        super().__init__(instance, config)

        self.solver_class = solver_class
        self.layout = self._build_layout()

    def _build_layout(self) -> Layout:
        """
        Constructs the Layout object based on config:
        - Uses explicit coordinates and blockade if provided.
        - Otherwise uses device information if use_quantum is True.
        - If use_quantum is False, requires rydberg_blockade for layout generation.

        Returns:
            Layout: The constructed layout.
        """
        if self.config.layout_coords is not None and self.config.rydberg_blockade is not None:
            return Layout(data=self.config.layout_coords, rydberg_blockade=self.config.rydberg_blockade)
        elif self.config.use_quantum:
            if self.config.device is not None:
                return Layout.from_device(data=self.instance, device=self.config.device)
            else:
                raise ValueError("When use_quantum = True, either layout_coords & rydberg_blockade "
                                "or a backend must be provided in config.")
        elif not self.config.use_quantum:
            return Layout(data=self.instance, rydberg_blockade=self.config.rydberg_blockade)

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Entry point for solving the full MISInstance using recursive greedy decomposition.

        Returns:
            Execution containing a list of optimal or near-optimal MIS solutions.
        """
        return self._solve_recursive(self.instance)

    def _solve_recursive(self, instance: MISInstance) -> Execution[list[MISSolution]]:
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
        if len(graph) <= self.config.exact_solving_threshold:
            solver = self.solver_class(instance, self.config)
            return solver.solve()

        mappings = self._generate_subgraphs(graph)
        best_solution: MISSolution | None = None

        for mapping in mappings:
            subgraph = graph.subgraph(mapping.keys())
            if len(subgraph) <= self.config.exact_solving_threshold:
                continue

            layout_subgraph = self._generate_layout_graph(graph, mapping)
            sub_instance = MISInstance(graph=layout_subgraph)
            solver = self.solver_class(sub_instance, self.config)

            results = solver.solve().result()[:self.config.max_number_of_solutions]
            inv_map = {v: k for k, v in mapping.items()}

            for partial in results:
                logical_nodes = [inv_map[n] for n in partial.nodes]
                reduced_graph = self._remove_nodes(graph, logical_nodes)
                remainder_instance = MISInstance(reduced_graph)

                remainder_exec = self._solve_recursive(remainder_instance)
                remainder_solutions = remainder_exec.result()

                for rem_sol in remainder_solutions:
                    combined_nodes = logical_nodes + rem_sol.nodes
                    energy = -calculate_weight(self.instance, combined_nodes)
                    combined_solution = MISSolution(original=graph, nodes=combined_nodes, energy=energy)

                    if (best_solution is None) or (energy < best_solution.energy):
                        best_solution = combined_solution

        if best_solution is None:
            return Execution.success([MISSolution(original=graph, nodes=[], energy=0)])
        return Execution.success([best_solution])

    def _generate_subgraphs(self, graph: nx.Graph) -> list[dict[int, int]]:
        """
        Generates subgraph mappings using greedy layout placement.

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
                previously_generated_subgraphs={}
            )
            mapping = mapper.generate(starting_node=node)
            mappings.append(mapping)
        return sorted(mappings, key=lambda m: len(m), reverse=True)[:self.config.subgraph_quantity]

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
            weight = graph.nodes[logical].get("weight", 1.0)
            pos = self.layout.graph.nodes[physical].get("pos", (0, 0))
            G.add_node(physical, weight=weight, pos=pos)

        for _, physical in mapping.items():
            for neighbor in self.layout.graph.neighbors(physical):
                if neighbor in mapping.values():
                    G.add_edge(physical, neighbor)

        return G

    def _remove_nodes(self, graph: nx.Graph, nodes: list[int]) -> nx.Graph:
        """
        Removes a node and all its neighbors from the graph.

        Args:
            graph: The graph to modify.
            nodes: List of nodes to remove.

        Returns:
            The reduced graph.
        """
        reduced = graph.copy()
        to_remove = set(nodes)
        for node in nodes:
            to_remove.update(graph.neighbors(node))
        reduced.remove_nodes_from(to_remove)
        return reduced
