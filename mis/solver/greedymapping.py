from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, List, Set, Tuple

import networkx as nx

from mis.shared.types import MISInstance
from mis.solver.lattice import Lattice

class GreedyMapping:
    """
    Performs a greedy mapping of a MISInstance's graph onto a physical lattice layout.
    """

    def __init__(
        self,
        instance: MISInstance,
        lattice: Lattice,
        previously_generated_subgraphs: List[Dict[int, int]],
        seed: int = 0,
    ) -> None:
        """
        Initializes the GreedyMapping algorithm for mapping a graph onto a lattice.

        Args:
            instance: The MIS problem instance containing the logical graph.
            lattice: The lattice structure defining the physical layout.
            previous_mappings: List of previous mappings (for scoring reuse).
            seed: Random seed for reproducibility.
        """
        self.graph: nx.Graph = instance.graph.copy()
        self.lattice_instance: Lattice = lattice
        self.lattice: nx.Graph = nx.convert_node_labels_to_integers(self.lattice_instance.lattice)
        self.previously_generated_subgraphs = previously_generated_subgraphs
        random.seed(seed)

    def generate(self,
                 starting_node: int,
        remove_invalid_placement_nodes: bool = True,
        rank_nodes: bool = True,
    ) -> Dict[int, int]:
        """
        Generates a subgraph by mapping the input graph onto the lattice using a greedy approach.

        Args:
            starting_node: The initial graph node to start mapping.
            remove_invalid_placement_nodes: Whether to remove invalid placements.
            rank_nodes: Whether to rank nodes using the scoring heuristic.

        Returns:
            dict: A dictionary representing the graph-to-lattice mapping.
        """
        unmapping: Dict[int, int] = {}
        mapping: Dict[int, int] = {}
        unexpanded_nodes: Set[int] = set()

        current_lattice_node: int = self._initialize(
            starting_node, mapping, unmapping, unexpanded_nodes
        )
        current_node: int = starting_node

        while unexpanded_nodes:
            unexpanded_nodes.remove(current_node)

            lattice_neighbors = list(self.lattice.neighbors(current_lattice_node))
            free_lattice_neighbors = [
                neighbor for neighbor in lattice_neighbors if neighbor not in unmapping
            ]

            neighbors = list(self.graph.neighbors(current_node))

            self._extend_mapping(
                considered_nodes=neighbors,
                unexpanded_nodes=unexpanded_nodes,
                free_lattice_neighbors=free_lattice_neighbors,
                mapping=mapping,
                unmapping=unmapping,
                remove_invalid_placement_nodes=remove_invalid_placement_nodes,
                rank_nodes=rank_nodes,
            )

            if unexpanded_nodes:
                current_node = next(iter(unexpanded_nodes))
                current_lattice_node = mapping[current_node]

        if not self._validate(mapping, unmapping):
            raise Exception("Invalid mapping!")

        return mapping

    def _initialize(
        self,
        starting_node: int,
        mapping: Dict[int, int],
        unmapping: Dict[int, int],
        unexpanded_nodes: Set[int],
    ) -> int:
        """Initializes mapping at the center of the lattice.
        
        Args:
            starting_node: The initial node in the graph.
            mapping: Dictionary for graph-to-lattice mapping.
            unmapping: Dictionary for lattice-to-graph mapping.
            unexpanded_nodes: Set of unexpanded nodes in the graph.

        Returns: 
            The lattice node corresponding to the starting node.
        """
        lattice_n: int = nx.number_of_nodes(self.lattice)
        lattice_grid_size: int = int(math.sqrt(lattice_n))
        starting_lattice_node: int = int(lattice_n / 2 + lattice_grid_size / 4)
        mapping[starting_node] = starting_lattice_node
        unmapping[starting_lattice_node] = starting_node
        unexpanded_nodes.add(starting_node)
        return starting_lattice_node

    def _extend_mapping(
        self,
        considered_nodes: List[int],
        unexpanded_nodes: Set[int],
        free_lattice_neighbors: List[int],
        mapping: Dict[int, int],
        unmapping: Dict[int, int],
        remove_invalid_placement_nodes: bool = True,
        rank_nodes: bool = True,
    ) -> None:
        """
        Extends the mapping by assigning unplaced graph nodes to free lattice nodes.

        Args:
            considered_nodes: Nodes in the graph being considered for mapping.
            unexpanded_nodes: Set of unexpanded nodes.
            free_lattice_neighbors: Available lattice neighbors for mapping.
            mapping: Current graph-to-lattice mapping.
            unmapping: Current lattice-to-graph mapping.
            remove_invalid_placement_nodes: Whether to remove invalid placements.
            rank_nodes: Whether to rank nodes using the scoring heuristic.
        """
        already_placed_nodes: Set[int] = set(mapping.keys())
        unplaced_nodes: List[int] = [
            n for n in considered_nodes if n not in already_placed_nodes
        ]

        if rank_nodes:
            node_scoring = self._score_nodes(
                unplaced_nodes, mapping, remove_invalid_placement_nodes
            )
            unplaced_nodes.sort(key=lambda n: node_scoring[n], reverse=True)

        for free_latt_neighbor in free_lattice_neighbors:
            for unplaced_node in unplaced_nodes:
                valid_placement: bool = True

                free_latt_neighbor_neighbors = list(self.lattice.neighbors(free_latt_neighbor))
                free_latt_neighbor_mapped_neighbors = [
                    n for n in free_latt_neighbor_neighbors if n in unmapping
                ]
                for mapped_neighbor in free_latt_neighbor_mapped_neighbors:
                    if not self.graph.has_edge(unplaced_node, unmapping[mapped_neighbor]):
                        valid_placement = False
                        break

                if valid_placement:
                    candidate_neighbors = list(self.graph.neighbors(unplaced_node))
                    for neighbor in candidate_neighbors:
                        if neighbor in already_placed_nodes and not self.lattice.has_edge(
                            mapping[neighbor], free_latt_neighbor
                        ):
                            valid_placement = False
                            break

                if valid_placement:
                    mapping[unplaced_node] = free_latt_neighbor
                    unmapping[free_latt_neighbor] = unplaced_node
                    already_placed_nodes.add(unplaced_node)
                    unplaced_nodes.remove(unplaced_node)
                    unexpanded_nodes.add(unplaced_node)
                    break

        if remove_invalid_placement_nodes:
            self.graph.remove_nodes_from(unplaced_nodes)

    def _score_nodes(
        self,
        nodes_to_score: List[int],
        mapping: Dict[int, int],
        remove_invalid_placement_nodes: bool,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Scores nodes for placement using a greedy heuristic.

        Args:
            nodes_to_score: List of nodes to score.
            mapping: Current graph-to-lattice mapping.
            remove_invalid_placement_nodes: Whether to penalize invalid placements.

        Returns:
            Dictionary mapping nodes to scores with random tiebreakers.
        """
        n: int = nx.number_of_nodes(self.graph)
        node_scores: Dict[int, float] = {}

        for node in nodes_to_score:
            degree_score: float = 1 - (
                abs(self.lattice_instance.avg_degree - self.graph.degree(node)) / n
            )
            non_adj_score: float = 0
            if not remove_invalid_placement_nodes:
                non_neighbors = [
                    neighbor for neighbor in nx.non_neighbors(self.graph, node) if neighbor in mapping
                ]
                if n > 0:
                    non_adj_score = len(non_neighbors) / n

            subgraphs_containing_node_count: int = sum(
                1 for subgraph in self.previously_generated_subgraphs if node in subgraph
            )
            previous_subgraphs_belonging_score: float = (
                1
                - (subgraphs_containing_node_count / len(self.previously_generated_subgraphs))
                if self.previously_generated_subgraphs
                else 0
            )

            node_scores[node] = degree_score + non_adj_score + previous_subgraphs_belonging_score

        return {node: (score, random.random()) for node, score in node_scores.items()}

    def _validate(
        self, mapping: Dict[int, int], unmapping: Dict[int, int]
    ) -> bool:
        """
        Checks if the current mapping is valid based on adjacency constraints.

        Args:
            mapping: Graph-to-lattice mapping.
            unmapping: Lattice-to-graph mapping.

        Return:
            True if the mapping is valid, False otherwise.
        """
        for node in self.graph.nodes():
            if node in mapping:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in mapping and not self.lattice.has_edge(
                        mapping[node], mapping[neighbor]
                    ):
                        return False

        for latt_node in self.lattice.nodes():
            if latt_node in unmapping:
                for latt_neighbor in self.lattice.neighbors(latt_node):
                    if latt_neighbor in unmapping and not self.graph.has_edge(
                        unmapping[latt_node], unmapping[latt_neighbor]
                    ):
                        return False

        return True
