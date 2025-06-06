from __future__ import annotations

import math
import random

import networkx as nx

from mis.shared.types import MISInstance
from mis.shared.error import MappingError
from mis.pipeline.layout import Layout


class GreedyMapping:
    """
    Performs a greedy mapping of a MISInstance's graph onto a physical layout.
    """

    def __init__(
        self,
        instance: MISInstance,
        layout: Layout,
        previous_subgraphs: list[dict[int, int]],
        seed: int = 0,
    ) -> None:
        """
        Initializes the GreedyMapping algorithm for mapping a graph onto a layout.

        Args:
            instance: The MIS problem instance containing the logical graph.
            layout: The layout structure defining the physical geometry.
            previous_subgraphs: list of previous mappings (for scoring reuse).
            seed: Random seed for reproducibility.
        """
        self.graph: nx.Graph = instance.graph.copy()
        self.layout_graph: nx.Graph = nx.convert_node_labels_to_integers(layout.graph)
        self.layout_avg_degree = layout.avg_degree
        self.previous_subgraphs = previous_subgraphs
        random.seed(seed)

    def generate(
        self,
        starting_node: int,
        remove_invalid_placement_nodes: bool = True,
        rank_nodes: bool = True,
    ) -> dict[int, int]:
        """
        Generates a subgraph by mapping the input graph onto the layout using a greedy approach.

        Args:
            starting_node: The initial graph node to start mapping.
            remove_invalid_placement_nodes: Whether to remove invalid placements.
            rank_nodes: Whether to rank nodes using the scoring heuristic.

        Returns:
            dict: A dictionary representing the graph-to-layout mapping.
        """
        # dictionary for graph-to-layout mapping.
        mapping: dict[int, int] = dict()
        # dictionary for layout-to-graph mapping.
        unmapping: dict[int, int] = dict()

        current_layout_node, unexpanded_nodes = self._initialize(
            self.layout_graph, starting_node, mapping, unmapping
        )
        current_node: int = starting_node

        while unexpanded_nodes:
            unexpanded_nodes.remove(current_node)

            layout_neighbors = list(self.layout_graph.neighbors(current_layout_node))
            free_layout_neighbors = [
                neighbor for neighbor in layout_neighbors if neighbor not in unmapping
            ]

            neighbors = list(self.graph.neighbors(current_node))

            self._extend_mapping(
                considered_nodes=neighbors,
                unexpanded_nodes=unexpanded_nodes,
                free_layout_neighbors=free_layout_neighbors,
                mapping=mapping,
                unmapping=unmapping,
                remove_invalid_placement_nodes=remove_invalid_placement_nodes,
                rank_nodes=rank_nodes,
            )

            if unexpanded_nodes:
                current_node = next(iter(unexpanded_nodes))
                current_layout_node = mapping[current_node]

        if not self._validate(mapping, unmapping):
            raise MappingError("Invalid mapping based on adjacency constraints.")

        return mapping

    @staticmethod
    def _initialize(
        layout_graph: nx.Graph,
        starting_node: int,
        mapping: dict[int, int],
        unmapping: dict[int, int],
    ) -> tuple[int, set[int]]:
        """Initializes mapping at the center of the layout.

        Args:
            layout_graph: Layout graph
            starting_node: The initial node in the graph.
            mapping: dictionary for graph-to-layout mapping.
            unmapping: dictionary for layout-to-graph mapping.

        Returns:
            tuple[int, set[int]]: tuple of
                - The layout node corresponding to the starting node.
                - set of unexpanded nodes in the graph.
        """
        layout_n: int = nx.number_of_nodes(layout_graph)
        layout_grid_size: int = int(math.sqrt(layout_n))
        starting_layout_node: int = int(layout_n / 2 + layout_grid_size / 4)
        mapping[starting_node] = starting_layout_node
        unmapping[starting_layout_node] = starting_node
        unexpanded_nodes: set[int] = set()
        unexpanded_nodes.add(starting_node)
        return starting_layout_node, unexpanded_nodes

    def _extend_mapping(
        self,
        considered_nodes: list[int],
        unexpanded_nodes: set[int],
        free_layout_neighbors: list[int],
        mapping: dict[int, int],
        unmapping: dict[int, int],
        remove_invalid_placement_nodes: bool = True,
        rank_nodes: bool = True,
    ) -> None:
        """
        Extends the mapping by assigning unplaced graph nodes to free layout nodes.

        Args:
            considered_nodes: Nodes in the graph being considered for mapping.
            unexpanded_nodes: set of unexpanded nodes.
            free_layout_neighbors: Available layout neighbors for mapping.
            mapping: Current graph-to-layout mapping.
            unmapping: Current layout-to-graph mapping.
            remove_invalid_placement_nodes: Whether to remove invalid placements.
            rank_nodes: Whether to rank nodes using the scoring heuristic.
        """
        already_placed_nodes: set[int] = set(mapping.keys())
        unplaced_nodes: list[int] = [n for n in considered_nodes if n not in already_placed_nodes]

        if rank_nodes:
            node_scoring = self._score_nodes(
                unplaced_nodes, mapping, remove_invalid_placement_nodes
            )
            unplaced_nodes.sort(key=lambda n: node_scoring[n], reverse=True)

        for free_layout_node in free_layout_neighbors:
            for unplaced_node in unplaced_nodes:
                valid_placement: bool = True

                layout_neighbors = list(self.layout_graph.neighbors(free_layout_node))
                mapped_neighbors = [n for n in layout_neighbors if n in unmapping]

                for mapped_neighbor in mapped_neighbors:
                    if not self.graph.has_edge(unplaced_node, unmapping[mapped_neighbor]):
                        valid_placement = False
                        break

                if valid_placement:
                    candidate_neighbors = list(self.graph.neighbors(unplaced_node))
                    for neighbor in candidate_neighbors:
                        if neighbor in already_placed_nodes and not self.layout_graph.has_edge(
                            mapping[neighbor], free_layout_node
                        ):
                            valid_placement = False
                            break

                if valid_placement:
                    mapping[unplaced_node] = free_layout_node
                    unmapping[free_layout_node] = unplaced_node
                    already_placed_nodes.add(unplaced_node)
                    unplaced_nodes.remove(unplaced_node)
                    unexpanded_nodes.add(unplaced_node)
                    break

        if remove_invalid_placement_nodes:
            self.graph.remove_nodes_from(unplaced_nodes)

    def _score_nodes(
        self,
        nodes_to_score: list[int],
        mapping: dict[int, int],
        remove_invalid_placement_nodes: bool,
    ) -> dict[int, tuple[float, float]]:
        """
        Scores nodes for placement using a greedy heuristic.

        Args:
            nodes_to_score: list of nodes to score.
            mapping: Current graph-to-layout mapping.
            remove_invalid_placement_nodes: Whether to penalize invalid placements.

        Returns:
            dictionary mapping nodes to scores with random tiebreakers.
        """
        n: int = nx.number_of_nodes(self.graph)
        node_scores: dict[int, float] = {}

        for node in nodes_to_score:
            degree_score: float = 1 - (abs(self.layout_avg_degree - self.graph.degree(node)) / n)
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
                1 for subgraph in self.previous_subgraphs if node in subgraph
            )
            previous_subgraphs_belonging_score: float = (
                1 - (subgraphs_containing_node_count / len(self.previous_subgraphs))
                if self.previous_subgraphs
                else 0
            )

            node_scores[node] = degree_score + non_adj_score + previous_subgraphs_belonging_score

        return {node: (score, random.random()) for node, score in node_scores.items()}

    def _validate(self, mapping: dict[int, int], unmapping: dict[int, int]) -> bool:
        """
        Checks if the current mapping is valid based on adjacency constraints.

        Args:
            mapping: Graph-to-layout mapping.
            unmapping: layout-to-graph mapping.

        Return:
            True if the mapping is valid, False otherwise.
        """
        for node in self.graph.nodes():
            if node in mapping:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in mapping and not self.layout_graph.has_edge(
                        mapping[node], mapping[neighbor]
                    ):
                        return False

        for layout_node in self.layout_graph.nodes():
            if layout_node in unmapping:
                for layout_neighbor in self.layout_graph.neighbors(layout_node):
                    if layout_neighbor in unmapping and not self.graph.has_edge(
                        unmapping[layout_node], unmapping[layout_neighbor]
                    ):
                        return False

        return True
