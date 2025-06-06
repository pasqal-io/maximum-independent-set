from __future__ import annotations

from statistics import mean

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mis.shared.types import MISInstance
from pulser.devices import Device


class Layout:
    """
    A 2D layout class for quantum layout embedding.

    Accepts either:
    - dict[int, tuple[float, float]] of coordinates
        mapping from node (int) to physical coordinates (x, y)
        UNIT = "µm"
    - MISInstance (graph)

    Uses a distance threshold (rydberg_blockade ("µm")) to create edges.
    """

    def __init__(
        self,
        data: MISInstance | dict[int, tuple[float, float]],
        rydberg_blockade: float,
    ):
        self.coords = self._get_coords(data)
        self.rydberg_blockade = rydberg_blockade
        self.graph = self._build_graph()
        self.avg_degree = self._compute_avg_degree()

    @staticmethod
    def _get_coords(data: MISInstance | dict[int, tuple[float, float]]) -> dict[int, np.ndarray]:
        """
        Get layout coordinates from either a MISInstance or a raw coordinate dictionary.

        If a MISInstance is given, use a spring layout to generate (x, y) positions.
        If a dictionary is given, return it unchanged.

        Args:
            data: A MISInstance or dict of coordinates.

        Returns:
            A dictionary mapping node IDs to (x, y) coordinates.
        """
        if isinstance(data, MISInstance):
            coords = nx.spring_layout(data.graph)
            return {int(k): np.array(v, dtype=float) for k, v in coords.items()}
        elif isinstance(data, dict):
            return {int(k): np.array(v, dtype=float) for k, v in data.items()}
        else:
            raise TypeError("Expected data to be MISInstance or dict[int, tuple[float, float]]")

    @classmethod
    def from_device(
        cls,
        data: MISInstance | dict[int, tuple[float, float]],
        device: Device,
    ) -> Layout:
        """
        Creates a Layout using `device.min_atom_distance` as the blockade,
        and rescales coordinates so no pair is too close.
        """
        coords = cls._get_coords(data)
        assert len(coords) >= 1

        # Compute all pairwise distances
        distances = [
            np.linalg.norm(coords[v1] - coords[v2]) for v1 in coords for v2 in coords if v1 < v2
        ]
        min_distance = min(distances)
        if min_distance < device.min_atom_distance:
            scale = device.min_atom_distance / min_distance
            coords = {k: tuple(v * scale) for k, v in coords.items()}
        else:
            coords = {k: tuple(v) for k, v in coords.items()}

        return cls(data=coords, rydberg_blockade=device.min_atom_distance)

    def _build_graph(self) -> nx.Graph:
        node_ids = list(self.coords.keys())
        positions = np.array([self.coords[node_id] for node_id in node_ids])  # shape: (n, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # shape: (n, n, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)  # shape: (n, n)

        G = nx.Graph()
        for node_id, pos in zip(node_ids, positions):
            G.add_node(node_id, pos=tuple(pos))

        # Add edges where distance < rydberg_blockade (exclude diagonal)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if dist_matrix[i, j] < self.rydberg_blockade:
                    G.add_edge(node_ids[i], node_ids[j])

        return G

    def _compute_avg_degree(self) -> int:
        degrees = [deg for _, deg in self.graph.degree()]
        return int(mean(degrees)) if degrees else 0

    def draw(self) -> None:
        pos = nx.get_node_attributes(self.graph, "pos")
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color="skyblue")
        plt.title("Layout Graph")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    def num_nodes(self) -> int:
        return int(self.graph.number_of_nodes())

    def grid_size(self) -> int:
        return int(round(self.num_nodes() ** 0.5))
