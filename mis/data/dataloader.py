import pandas as pd
import networkx as nx
from mis import MISInstance
from geopy.distance import geodesic
from pathlib import Path


class DataLoader:
    """
    DataLoader class to handle loading of coordinates from CSV files,
    calculating distances between coordinates, and building a Maximum Independent Set (MIS) instance.

    Attributes:
        coordinates_dataset (list[tuple[float, float]]): A list of tuples representing coordinates (latitude, longitude).
    """

    coordinates_dataset: list[tuple[float, float]]

    @staticmethod
    def distance_from_coordinates(
        coord1: tuple[float, float], coord2: tuple[float, float]
    ) -> float:
        """
        Calculate the distance between two geodesic coordinates.

        Args:
            coord1 (tuple[float, float]): The first coordinate as a tuple (latitude, longitude).
            coord2 (tuple[float, float]): The second coordinate as a tuple (latitude, longitude).

        Returns:
            float: The distance between the two coordinates in kilometers.
        """
        return float(geodesic(coord1, coord2).km)

    def load_from_csv_coordinates(self, file_path: Path) -> None:
        """
        Load coordinates from a CSV file.
        The CSV file should have a column named 'coordonnees' with coordinates in the format "lat,lon".

        Args:
            file_path (Path): The path to the CSV file containing coordinates.
        """
        # error handling with a try-except block
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        if not file_path.is_file():
            raise ValueError(f"The path {file_path} is not a file.")
        if not file_path.suffix == ".csv":
            raise ValueError(f"The file {file_path} is not a CSV file.")

        df = pd.read_csv(file_path, sep=";")

        if "coordonnees" not in df.columns:
            raise ValueError("The CSV file must contain a 'coordonnees' column.")
        if df.empty:
            raise ValueError("The CSV file is empty.")
        if df["coordonnees"].isnull().any():
            raise ValueError("The 'coordonnees' column contains null values.")
        if not all(df["coordonnees"].apply(lambda x: len(x.split(",")) == 2)):
            raise ValueError(
                "All entries in the 'coordonnees' column must be in the format 'lat,lon'."
            )

        self.coordinates_dataset = [
            (float(c.split(",")[0]), float(c.split(",")[1])) for c in df["coordonnees"]
        ]

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
        Returns:
            MISInstance: An instance of the Maximum Independent Set problem represented as a graph.
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
