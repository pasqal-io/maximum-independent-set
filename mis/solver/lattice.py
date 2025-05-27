from __future__ import annotations

import random
from statistics import mean

import networkx as nx
import matplotlib.pyplot as plt


class Lattice:
    """
    Initializes the lattice structure with given coordinates and Rydberg blockade distance.
    Represents a 2D lattice structure for quantum embedding,
    where nodes are connected if they are within a given Rydberg blockade radius.
    """

    def __init__(
        self,
        lattice_coords: dict[int, tuple[float, float]],
        rydberg_blockade: float,
        seed: int = 0,
    ) -> None:
        """
        Args:
            lattice_coords: Mapping from lattice node ID to (x, y) coordinates.
            rydberg_blockade: Max interaction distance (used for edge generation).
            seed: Optional random seed.
        """
        self.lattice_coords = lattice_coords
        self.rydberg_blockade = rydberg_blockade
        self.seed = seed
        random.seed(seed)

        self.lattice = self._build_graph()
        self.avg_degree = self._compute_avg_degree()

    def _build_graph(self) -> nx.Graph:
        """Creates a lattice graph where edges connect nodes within the blockade radius."""
        G = nx.Graph()
        for id, point_coord in self.lattice_coords.items():
            G.add_node(id, pos=(point_coord[0], point_coord[1]))

        for id1, coord1 in self.lattice_coords.items():
            for id2, coord2 in self.lattice_coords.items():
                if id1 < id2:
                    distance: float = (
                        (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
                    ) ** 0.5
                    if distance < self.rydberg_blockade:
                        G.add_edge(id1, id2)

        return G

    def _compute_avg_degree(self) -> int:
        degrees = [deg for _, deg in self.lattice.degree()]
        return int(mean(degrees)) if degrees else 0
    
    def num_nodes(self) -> int:
        """Returns the number of nodes in the lattice."""
        return nx.number_of_nodes(self.lattice)

    def grid_size(self) -> int:
        """
        Estimates the grid size assuming a roughly square lattice.
        Useful when nodes are laid out in a square-like arrangement.

        Returns:
            An integer representing the approximate width/height of the lattice grid.
        """
        return int(round(self.num_nodes() ** 0.5))

    def display(self) -> None:
        """Draws the lattice with positions and edges."""
        pos = nx.get_node_attributes(self.lattice, "pos")
        plt.figure(figsize=(8, 6))
        nx.draw(self.lattice, pos, with_labels=True, node_size=500, node_color="skyblue")
        plt.title("Lattice Graph")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

