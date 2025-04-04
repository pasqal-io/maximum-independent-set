import copy
import math
import random
from statistics import mean
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class Lattice:
    def __init__(
        self,
        lattice_coords_dic: Dict[int, Tuple[float, float]],
        rydberg_blockade: float,
        seed: int = 0,
    ) -> None:
        """
        Initializes the lattice structure with given coordinates and Rydberg blockade distance.

        :param lattice_coords_dic: Dictionary mapping lattice node IDs to their coordinates.
        :param rydberg_blockade: The maximum distance for which two nodes are considered neighbors.
        :param seed: Random seed for reproducibility.
        """
        random.seed(seed)
        self.lattice_coords_dic: Dict[int, Tuple[float, float]] = lattice_coords_dic
        self.rydberg_blockade: float = rydberg_blockade
        self.lattice: nx.Graph = self.genereGraphFromCoords()

        degrees: List[int] = [self.lattice.degree(node) for node in self.lattice.nodes()]
        self.avg_degree: int = int(mean(degrees))

    def genereGraphFromCoords(self) -> nx.Graph:
        """
        Generates a graph based on lattice coordinates and Rydberg blockade constraints.

        :return: A NetworkX graph representing the lattice.
        """
        G: nx.Graph = nx.Graph()

        for id, point_coord in self.lattice_coords_dic.items():
            G.add_node(id, pos=(point_coord[0], point_coord[1]))

        for id1, coord1 in self.lattice_coords_dic.items():
            for id2, coord2 in self.lattice_coords_dic.items():
                if id1 < id2:
                    distance: float = (
                        (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
                    ) ** 0.5
                    if distance < self.rydberg_blockade:
                        G.add_edge(id1, id2)
        return G

    def display_lattice(self) -> None:
        """
        Visualizes the lattice graph with nodes and edges.
        """
        positions: Dict[int, Tuple[float, float]] = nx.get_node_attributes(self.lattice, "pos")

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.lattice,
            pos=positions,
            with_labels=True,
            node_size=700,
            node_color="lightgreen",
        )
        plt.title(f"Lattice graph (rydberg blockade = {self.rydberg_blockade})")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()


class GreedyMapping:
    def __init__(
        self,
        graph: nx.Graph,
        lattice_instance: Lattice,
        previously_generated_subgraphs: List[Dict[int, int]],
        seed: int = 0,
    ) -> None:
        """
        Initializes the GreedyMapping algorithm for mapping a graph onto a lattice.

        :param graph: The input graph to map.
        :param lattice_instance: The lattice instance used for mapping.
        :param previously_generated_subgraphs: List of previously generated subgraphs for scoring.
        :param seed: Random seed for reproducibility.
        """
        random.seed(seed)
        self.graph: nx.Graph = copy.deepcopy(graph)
        self.lattice_instance: Lattice = lattice_instance
        self.lattice: nx.Graph = nx.convert_node_labels_to_integers(lattice_instance.lattice)
        self.previously_generated_subgraphs: List[Dict[int, int]] = previously_generated_subgraphs

    def initialize_mapping(
        self,
        starting_node: int,
        mapping: Dict[int, int],
        unmapping: Dict[int, int],
        unexpanded_nodes: Set[int],
    ) -> int:
        """
        Initializes the mapping process by selecting the center of the lattice as the starting point.

        :param starting_node: The initial node in the graph.
        :param mapping: Dictionary for graph-to-lattice mapping.
        :param unmapping: Dictionary for lattice-to-graph mapping.
        :param unexpanded_nodes: Set of unexpanded nodes in the graph.
        :return: The lattice node corresponding to the starting node.
        """
        lattice_n: int = nx.number_of_nodes(self.lattice)
        lattice_grid_size: int = int(math.sqrt(lattice_n))
        starting_lattice_node: int = int(lattice_n / 2 + lattice_grid_size / 4)
        mapping[starting_node] = starting_lattice_node
        unmapping[starting_lattice_node] = starting_node
        unexpanded_nodes.add(starting_node)
        return starting_lattice_node

    def check_mapping_validity(self, mapping: Dict[int, int], unmapping: Dict[int, int]) -> bool:
        """
        Checks if the current mapping is valid based on adjacency constraints.

        :param mapping: Graph-to-lattice mapping.
        :param unmapping: Lattice-to-graph mapping.
        :return: True if the mapping is valid, False otherwise.
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

    def greedy_node_scoring(
        self,
        nodes_to_score: List[int],
        mapping: Dict[int, int],
        remove_invalid_placement_nodes: bool,
    ) -> Dict[int, Tuple[float, float]]:
        """
        Scores nodes for placement using a greedy heuristic.

        :param nodes_to_score: List of nodes to score.
        :param mapping: Current graph-to-lattice mapping.
        :param remove_invalid_placement_nodes: Whether to penalize invalid placements.
        :return: Dictionary mapping nodes to scores with random tiebreakers.
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
                    neighbor
                    for neighbor in nx.non_neighbors(self.graph, node)
                    if neighbor in mapping
                ]
                if n > 0:
                    non_adj_score = len(non_neighbors) / n

            subgraphs_containing_node_count: int = sum(
                1 for subgraph in self.previously_generated_subgraphs if node in subgraph
            )
            previous_subgraphs_belonging_score: float = (
                1 - (subgraphs_containing_node_count / len(self.previously_generated_subgraphs))
                if self.previously_generated_subgraphs
                else 0
            )

            node_scores[node] = degree_score + non_adj_score + previous_subgraphs_belonging_score

        return {node: (score, random.random()) for node, score in node_scores.items()}

    def extend_mapping_with_nodes(
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

        :param considered_nodes: Nodes in the graph being considered for mapping.
        :param unexpanded_nodes: Set of unexpanded nodes.
        :param free_lattice_neighbors: Available lattice neighbors for mapping.
        :param mapping: Current graph-to-lattice mapping.
        :param unmapping: Current lattice-to-graph mapping.
        :param remove_invalid_placement_nodes: Whether to remove invalid placements.
        :param rank_nodes: Whether to rank nodes using the scoring heuristic.
        """
        already_placed_nodes: Set[int] = set(mapping.keys())
        unplaced_nodes: List[int] = [n for n in considered_nodes if n not in already_placed_nodes]

        if rank_nodes:
            node_scoring = self.greedy_node_scoring(
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

    def generate_greedy_ud_subgraph_with(
        self,
        starting_node: int,
        remove_invalid_placement_nodes: bool = True,
        rank_nodes: bool = True,
    ) -> Dict[int, int]:
        """
        Generates a subgraph by mapping the input graph onto the lattice using a greedy approach.

        :param starting_node: The initial graph node to start mapping.
        :param remove_invalid_placement_nodes: Whether to remove invalid placements.
        :param rank_nodes: Whether to rank nodes using the scoring heuristic.
        :return: A dictionary representing the graph-to-lattice mapping.
        """
        unmapping: Dict[int, int] = {}
        mapping: Dict[int, int] = {}
        unexpanded_nodes: Set[int] = set()

        current_lattice_node: int = self.initialize_mapping(
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

            self.extend_mapping_with_nodes(
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

        if not self.check_mapping_validity(mapping, unmapping):
            raise Exception("Invalid mapping!")

        return mapping
