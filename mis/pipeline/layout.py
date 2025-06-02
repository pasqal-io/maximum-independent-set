from __future__ import annotations

from typing import Union
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
    - MISInstance (graph)

    Uses a distance threshold (rydberg_blockade) to create edges.
    """

    def __init__(
        self,
        data: Union[MISInstance, dict[int, tuple[float, float]]],
        rydberg_blockade: float,
    ):
        if isinstance(data, MISInstance):
            coords = nx.spring_layout(data.graph)
            coords = {int(k): tuple(map(float, v)) for k, v in coords.items()}
        elif isinstance(data, dict):
            coords = data
        else:
            raise TypeError("Expected data to be MISInstance or dict[int, tuple[float, float]]")

        self.coords = coords
        self.rydberg_blockade = rydberg_blockade
        self.graph = self._build_graph()
        self.avg_degree = self._compute_avg_degree()

    @classmethod
    def from_device(
        cls,
        data: Union[MISInstance, dict[int, tuple[float, float]]],
        device: Device,
    ) -> Layout:
        """
        Creates a Layout using `device.min_atom_distance` as the blockade,
        and rescales coordinates so no pair is too close.
        """
        if isinstance(data, MISInstance):
            coords = nx.spring_layout(data.graph)
            coords = {int(k): np.array(v, dtype=float) for k, v in coords.items()}
        elif isinstance(data, dict):
            coords = {int(k): np.array(v, dtype=float) for k, v in data.items()}
        else:
            raise TypeError("Expected data to be MISInstance or dict[int, tuple[float, float]]")

        # Compute all pairwise distances
        distances = [
            np.linalg.norm(coords[v1] - coords[v2]) for v1 in coords for v2 in coords if v1 < v2
        ]

        if distances:
            min_distance = min(distances)
            if min_distance < device.min_atom_distance:
                scale = device.min_atom_distance / min_distance
                coords = {k: tuple(v * scale) for k, v in coords.items()}
            else:
                coords = {k: tuple(v) for k, v in coords.items()}
        else:
            coords = {k: tuple(v) for k, v in coords.items()}

        return cls(data=coords, rydberg_blockade=device.min_atom_distance)

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for node_id, position in self.coords.items():
            G.add_node(node_id, pos=position)

        node_items = list(self.coords.items())
        for i in range(len(node_items)):
            id1, coord1 = node_items[i]
            for j in range(i + 1, len(node_items)):
                id2, coord2 = node_items[j]
                distance = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5
                if distance < self.rydberg_blockade:
                    G.add_edge(id1, id2)
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

    def get_graph(self) -> nx.Graph:
        return self.graph

    def get_coords(self) -> dict[int, tuple[float, float]]:
        return self.coords  # type: ignore[no-any-return]
