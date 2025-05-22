from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import networkx as nx

from mis.shared.types import MISInstance, MISSolution
from mis.pipeline.execution import Execution
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.config import SolverConfig
from mis.solver.greedymapping import GreedyMapping
from mis.solver.lattice import Lattice


class GreedyMISSolver(BaseSolver):
    def __init__(
        self,
        instance: MISInstance,
        solver: BaseSolver,
    ) -> None:
        super().__init__(instance, solver.config)

        self.original_instance = instance
        self.sub_solver = solver
        self.config: SolverConfig = solver.config
        self.seed = 0

        self.lattice = Lattice(
            lattice_coords=self.config.lattice_coords,
            rydberg_blockade=self.config.rydberg_blockade,
            seed=self.seed,
        )

    def solve(self) -> Execution[List[MISSolution]]:
        return self._solve_recursive(self.original_instance)

    def _solve_recursive(self, instance: MISInstance) -> Execution[List[MISSolution]]:
        graph = instance.graph
        if len(graph) <= self.config.exact_solving_threshold:
            return self._exact_mis(instance)

        mappings = self._generate_subgraphs(graph)
        best_solution: MISSolution | None = None

        for mapping in mappings:
            subgraph = graph.subgraph(mapping.keys())
            if len(subgraph) <= self.config.exact_solving_threshold:
                continue

            lattice_subgraph = self._generate_lattice_graph(graph, mapping)
            sub_instance = MISInstance(graph=lattice_subgraph)
            sub_solver = type(self.sub_solver)(sub_instance, self.config)

            results = sub_solver.solve().result()[:self.config.max_number_of_solutions]
            inv_map = {v: k for k, v in mapping.items()}

            for partial in results:
                logical_nodes = [inv_map[n] for n in partial.nodes]
                reduced_graph = self._remove_nodes(graph, logical_nodes)
                remainder_instance = MISInstance(reduced_graph)

                remainder_exec = self._solve_recursive(remainder_instance)
                remainder_solutions = remainder_exec.result()

                for rem_sol in remainder_solutions:
                    combined_nodes = logical_nodes + rem_sol.nodes
                    energy = -self._calculate_weight(combined_nodes)
                    combined_solution = MISSolution(original=graph, nodes=combined_nodes, energy=energy)

                    if (best_solution is None) or (energy < best_solution.energy):
                        best_solution = combined_solution

        if best_solution is None:
            return Execution.success([MISSolution(original=graph, nodes=[], energy=0)])
        return Execution.success([best_solution])

    def _exact_mis(self, instance: MISInstance) -> Execution[List[MISSolution]]:
        graph = instance.graph
        best: List[int] = []
        self._backtrack(graph, [], 0, [best])
        energy = -self._calculate_weight(best)
        solution = MISSolution(original=graph, nodes=best, energy=energy)
        return Execution.success([solution])

    def _backtrack(self, graph: nx.Graph, subset: List[int], index: int, best: List[List[int]]) -> None:
        nodes = list(graph.nodes())
        if self._is_independent(graph, subset) and self._calculate_weight(subset) > self._calculate_weight(best[0]):
            best[0] = subset[:]
        for i in range(index, len(nodes)):
            subset.append(nodes[i])
            self._backtrack(graph, subset, i + 1, best)
            subset.pop()

    def _generate_subgraphs(self, graph: nx.Graph) -> List[Dict[int, int]]:
        mappings = []
        for node in graph.nodes():
            mapper = GreedyMapping(
                instance=MISInstance(graph),
                lattice=self.lattice,
                previously_generated_subgraphs=mappings,
                seed=self.seed,
            )
            mapping = mapper.generate(starting_node=node)
            mappings.append(mapping)
        return sorted(mappings, key=lambda m: len(m), reverse=True)[:self.config.subgraph_quantity]

    def _generate_lattice_graph(self, graph: nx.Graph, mapping: Dict[int, int]) -> nx.Graph:
        G = nx.Graph()
        for logical, physical in mapping.items():
            weight = graph.nodes[logical].get("weight", 1.0)
            pos = self.lattice.lattice.nodes[physical].get("pos", (0, 0))
            G.add_node(physical, weight=weight, pos=pos)

        for _, physical in mapping.items():
            for neighbor in self.lattice.lattice.neighbors(physical):
                if neighbor in mapping.values():
                    G.add_edge(physical, neighbor)

        return G

    def _remove_nodes(self, graph: nx.Graph, nodes: List[int]) -> nx.Graph:
        reduced = graph.copy()
        to_remove = set(nodes)
        for node in nodes:
            to_remove.update(graph.neighbors(node))
        reduced.remove_nodes_from(to_remove)
        return reduced

    def _calculate_weight(self, nodes: List[int]) -> float:
        return sum(self.original_instance.graph.nodes[n].get("weight", 1.0) for n in nodes)

    def _is_independent(self, graph: nx.Graph, nodes: List[int]) -> bool:
        return not any(graph.has_edge(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:])
