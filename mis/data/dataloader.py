"""This module provides functionality to load data from CSV files containing coordinates,
calculate distances between coordinates, and build a Maximum Independent Set (MIS) instance
from those coordinates. It uses the Haversine formula to compute distances and NetworkX to
construct the graph representation of the MIS problem."""

import pandas as pd
import networkx as nx
from mis import MISInstance
from geopy.distance import geodesic
from pathlib import Path


class DataLoader:
    """DataLoader class to handle loading of coordinates from CSV files,
    calculating distances between coordinates, and building a Maximum Independent Set (MIS) instance.
    """

    coordinates_dataset: list[tuple[float, float]]

    @staticmethod
    def distance_from_coordinates(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
        """Calculate the distance between two coordinates."""
        return float(geodesic(a_1, a_2).km)

    def load_from_csv_coordinates(self, file_path: Path) -> list[tuple[float, float]]:
        """
        Load coordinates from a CSV file and return them as a list of tuples.
        The CSV file should have a column named 'coordonnees' with coordinates in the format "lat,lon".
        """
        df = pd.read_csv(file_path, sep=";")
        self.coordinates_dataset = [
            (float(c.split(",")[0]), float(c.split(",")[1])) for c in df["coordonnees"]
        ]
        return self.coordinates_dataset

    def build_mis_instance_from_coordinates(
        self, antenna_range: float, antennas: set[int] = None
    ) -> MISInstance:
        """
        Build a Maximum Independent Set (MIS) instance from the loaded coordinates.
        The function creates a graph where nodes represent antennas and edges represent
        connections between antennas that are within the specified range.
        Args:
            antenna_range (float): The maximum distance between antennas to consider them connected.
            antennas (set[int], optional): A set of indices representing the antennas to include in the graph.
                                           If None, all antennas in the dataset are included.
        """
        if self.coordinates_dataset is None:
            raise ValueError(
                "Coordinates dataset is not loaded. Please load the dataset using load_from_csv_coordinates method."
            )

        if antennas is None:
            antennas = set(range(len(self.coordinates_dataset)))

        graph = nx.Graph()
        for i in antennas:
            graph.add_node(i)

        for i in antennas:
            for j in antennas:
                if (
                    i < j
                    and self.distance_from_coordinates(
                        self.coordinates_dataset[i], self.coordinates_dataset[j]
                    )
                    <= antenna_range
                ):
                    graph.add_edge(i, j)

        return MISInstance(graph)
