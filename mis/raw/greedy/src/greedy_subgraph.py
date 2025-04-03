import copy
import math
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import networkx as nx
from networkx import Graph
from greedy_lattice_mapping import GreedyMapping, Lattice
from classic_MIS_solvers import solve_weighted_mis, weighted_greedy_independent_set, weighted_generate_different_mis

class greedy_subgraph_solver:
    def __init__(
        self,
        graph: Graph,
        lattice_id_coord_dic: Dict[int, Tuple[float, float]],
        rydberg_blockade: float,
        mis_solving_function: Callable[[Graph, int], List[Set[int]]],
        seed: int = 0,
    ) -> None:
        """
        Initializes the greedy_subgraph_solver with the input graph, lattice parameters,
        and a function for solving the maximum independent set (MIS) problem.

        :param graph: The input graph.
        :param lattice_id_coord_dic: Dictionary mapping lattice IDs to coordinates.
        :param rydberg_blockade: The Rydberg blockade radius.
        :param mis_solving_function: Function to solve the MIS problem on a given graph.
        :param seed: Seed for randomness.
        """
        self.lattice: Lattice = Lattice(
            lattice_id_coord_dic,
            rydberg_blockade,
            seed=seed,
        )
        self.lattice_id_coord_dic: Dict[int, Tuple[float, float]] = lattice_id_coord_dic
        self.graph: Graph = graph
        self.mis_solving_function: Callable[[Graph, int], List[Set[int]]] = mis_solving_function

    def obtain_embeddable_subgraphs(
        self,
        current_graph: Graph,
        subgraph_quantity: int,
    ) -> List[Dict[int, int]]:
        """
        Generates a list of embeddable subgraphs based on greedy mapping, sorted by size.

        :param current_graph: The input graph from which subgraphs are extracted.
        :param subgraph_quantity: Number of largest subgraphs to return.
        :return: List of mappings representing subgraphs.
        """
        mappings: List[Dict[int, int]] = []
        for node in current_graph.nodes():
            greedy_mapper: GreedyMapping = GreedyMapping(
                current_graph,
                copy.deepcopy(self.lattice),
                {},
            )
            subgraph_mapping: Dict[int, int] = greedy_mapper.generate_greedy_ud_subgraph_with(node)
            mappings.append(subgraph_mapping)

        return sorted(mappings, key=lambda x: len(x), reverse=True)[:subgraph_quantity]

    def remove_mis_open_neighborhood_nodes_from_graph(
        self,
        current_graph: Graph,
        mis: List[int],
    ) -> Graph:
        """
        Removes MIS nodes and their open neighborhood from the graph.

        :param current_graph: The input graph.
        :param mis: List of nodes in the MIS.
        :return: A new graph with MIS nodes and their neighbors removed.
        """
        new_subgraph_with_removed_nodes: Graph = copy.deepcopy(current_graph)
        nodes_to_remove: Set[int] = set(mis)
        for node in mis:
            current_node_neighbors: Set[int] = set(current_graph.neighbors(node))
            nodes_to_remove.update(current_node_neighbors)

        new_subgraph_with_removed_nodes.remove_nodes_from(nodes_to_remove)
        return new_subgraph_with_removed_nodes

    def calculate_weight(self, node_list: List[int]) -> float:
        """
        Calculates the total weight of a given list of nodes in the graph.

        :param node_list: List of nodes.
        :return: Total weight of the nodes.
        """
        total_weight: float = 0
        for node in node_list:
            if "weight" in self.graph.nodes[node]:
                total_weight += self.graph.nodes[node]["weight"]
            else:
                raise ValueError(f"Node {node} does not have a 'weight' attribute.")
        return total_weight

    def is_independent_set(self, graph: Graph, subset: List[int]) -> bool:
        """
        Checks if the given subset of nodes is an independent set in the graph.

        :param graph: The input graph.
        :param subset: The list of nodes to check.
        :return: True if the subset is an independent set, False otherwise.
        """
        return not any(
            graph.has_edge(subset[i], subset[j])
            for i in range(len(subset))
            for j in range(i + 1, len(subset))
        )

    def backtracking_independent_set(
        self,
        graph: Graph,
        subset: List[int],
        index: int,
        max_set: List[List[int]],
    ) -> None:
        """
        Uses backtracking to find the maximum weighted independent set.

        :param graph: The input graph.
        :param subset: The current subset of nodes being explored.
        :param index: The current index in the graph nodes.
        :param max_set: A list containing the current maximum independent set (by reference).
        """
        nodes: List[int] = list(graph.nodes())
        current_weight: float = self.calculate_weight(subset)
        max_weight: float = self.calculate_weight(max_set[0])

        if self.is_independent_set(graph, subset) and current_weight > max_weight:
            max_set[0] = subset[:]

        for i in range(index, len(nodes)):
            subset.append(nodes[i])
            self.backtracking_independent_set(graph, subset, i + 1, max_set)
            subset.pop()

    def find_maximum_independent_set(self, graph: Graph) -> List[int]:
        """
        Finds the maximum weighted independent set in the graph using backtracking.

        :param graph: The input graph with weighted nodes.
        :return: A list of nodes representing the maximum weighted independent set.
        """
        max_set: List[List[int]] = [[]]
        self.backtracking_independent_set(graph, [], 0, max_set)
        return max_set[0]

    def generate_graph_to_solve(
        self,
        current_graph: Graph,
        lattice: Graph,
        mapping: Dict[int, int],
    ) -> Graph:
        """
        Generates a new subgraph to solve based on a mapping between the current graph
        and the lattice.

        :param current_graph: The input graph.
        :param lattice: The lattice graph.
        :param mapping: Mapping of nodes from the input graph to the lattice.
        :return: The generated subgraph.
        """
        subgraph: Graph = nx.Graph()
        for source_node, target_node in mapping.items():
            current_weight: Union[float, None] = current_graph.nodes[source_node].get("weight")
            lattice_pos: Union[Tuple[float, float], None] = lattice.nodes[target_node].get("pos")
            subgraph.add_node(
                target_node,
                weight=current_weight,
                pos=lattice_pos,
            )
        for source_node, target_node in mapping.items():
            for neighbor in lattice.neighbors(target_node):
                if neighbor in mapping.values():
                    subgraph.add_edge(target_node, neighbor)

        return subgraph

    def solve_recursively(
        self,
        current_graph: Graph,
        exact_solving_threshold: int,
        subgraph_quantity: int,
        mis_sample_quantity: int,
    ) -> List[int]:
        """
        Recursively solves the graph for the maximum independent set using a hybrid
        approach combining greedy subgraph extraction and exact solving.

        :param current_graph: The graph to solve.
        :param exact_solving_threshold: Size threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs to consider.
        :param mis_sample_quantity: Number of MIS samples to compute.
        :return: The maximum weighted independent set.
        """
        if len(current_graph) <= exact_solving_threshold:
            return self.find_maximum_independent_set(current_graph)

        mappings: List[Dict[int, int]] = self.obtain_embeddable_subgraphs(
            current_graph,
            subgraph_quantity,
        )
        current_max_mis: List[int] = []

        for mapping in mappings:
            nx_subgraph: Graph = current_graph.subgraph(mapping.keys())
            if nx_subgraph.number_of_nodes() <= exact_solving_threshold:
                self.solve_recursively(
                    nx_subgraph,
                    exact_solving_threshold,
                    subgraph_quantity,
                    mis_sample_quantity,
                )
                continue

            graph_to_solve: Graph = self.generate_graph_to_solve(
                current_graph,
                self.lattice.lattice,
                mapping,
            )

            current_mis_set_on_lattice: List[Set[int]] = self.mis_solving_function(
                     graph_to_solve,
                     self.lattice_id_coord_dic,
                     mis_sample_quantity,
                     )
         
            inverse_mapping: Dict[int, int] = {v: k for k, v in mapping.items()}
            current_mis_set: List[List[int]] = [
                [inverse_mapping[value] for value in mis_lattice]
                for mis_lattice in current_mis_set_on_lattice
            ]
            for current_mis in current_mis_set:
                new_subgraph_with_removed_nodes: Graph = (
                    self.remove_mis_open_neighborhood_nodes_from_graph(
                        current_graph,
                        current_mis,
                    )
                )
                #print(current_mis_set)
                #print('subgraph:', new_subgraph_with_removed_nodes)
                #print('weight:', nx.get_node_attributes(new_subgraph_with_removed_nodes, 'weight'))
                mis_from_recursive_call: List[int] = self.solve_recursively(
                    new_subgraph_with_removed_nodes,
                    exact_solving_threshold,
                    subgraph_quantity,
                    mis_sample_quantity,
                )
                if (
                    self.calculate_weight(current_mis)
                    + self.calculate_weight(mis_from_recursive_call)
                    > self.calculate_weight(current_max_mis)
                ):
                    current_max_mis = current_mis + mis_from_recursive_call

        if not self.is_independent_set(current_graph, current_max_mis):
            raise Exception("Not an independent set!")

        return current_max_mis

    def solve(
        self,
        exact_solving_threshold: int = 10,
        subgraph_quantity: int = 5,
        mis_sample_quantity: int = 1,
    ) -> List[int]:
        """
        Solves the maximum independent set problem on the input graph.

        :param exact_solving_threshold: Size threshold for exact solving.
        :param subgraph_quantity: Number of subgraphs to consider.
        :param mis_sample_quantity: Number of MIS samples to compute.
        :return: The maximum weighted independent set.
        """
        return self.solve_recursively(
            self.graph,
            exact_solving_threshold,
            subgraph_quantity,
            mis_sample_quantity,
        )