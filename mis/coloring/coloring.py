from copy import deepcopy
from typing import Optional
from mis import MISSolver
from mis.data.dataloader import DataLoader
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.config import SolverConfig
import matplotlib.pyplot as plt

from mis.shared.types import MISSolution


class GraphColoringSolver(BaseSolver):
    """
    GraphColoringSolver class to solve the graph coloring problem for antennas
    using the Maximum Independent Set (MIS) approach.
    Given the coordinates of antennas and a specified antenna range,
    it finds a coloring of the graph such that no two antennas in the range
    of each other share the same color.

    Attributes:
        loader (DataLoader): An instance of DataLoader to load antenna coordinates.
        antenna_range (float): The maximum distance within which antennas can interfere with each other.
        colors (list[int]): A list where the index represents the antenna and the value represents its color.
        colors_count (int): The total number of colors used in the solution.
        solver_config (SolverConfig): Configuration for the MIS solver, including backend and other settings.
    """

    loader: DataLoader
    antenna_range: float
    colors: list[int]
    colors_count: int
    solver_config: SolverConfig

    def __init__(
        self, loader: DataLoader, antenna_range: float, config: SolverConfig = SolverConfig()
    ):
        """
        Initialize the GraphColoringSolver with a DataLoader instance and antenna range.
        Args:
            loader (DataLoader): An instance of DataLoader to load antenna coordinates.
            antenna_range (float): The maximum distance within which antennas can interfere with each other.
            config (SolverConfig): Configuration for the MIS solver, including backend and other settings.
        """
        self.loader = loader
        self.antenna_range = antenna_range
        self.solver_config = config

    def solve(
        self, antennas: Optional[set[int]] = None, is_second_coloring: bool = False
    ) -> list[MISSolution]:
        """
        Solve the graph coloring problem by finding a maximum independent set
        for the given antenna range and coloring the antennas accordingly.

        Args:
            antennas (set[int]): A set of antenna indices to consider for coloring.
                If empty, all antennas in the dataset will be considered.
            is_second_coloring (bool): If True, the solver will not reset the colors count
                and will continue coloring from the last used color.

        Returns:
            Execution[list[MISSolution]]: An execution object containing the nodes of each color in the solution.
        """
        if antennas is None:
            antennas = set([x for x in range(len(self.loader.coordinates_dataset))])

        res = []
        if not is_second_coloring:
            self.colors = [-1] * len(self.loader.coordinates_dataset)
            self.colors_count = 0

        while len(antennas) > 0:
            solver = MISSolver(
                self.loader.build_mis_instance_from_coordinates(self.antenna_range, antennas),
                self.solver_config,
            )
            solutions = solver.solve()
            res.append(solutions[0])
            for antenna in solutions[0].nodes:
                self.colors[antenna] = self.colors_count
            antennas = antennas - set(solutions[0].nodes)
            self.colors_count += 1

        return res

    # split antennas into independent sets based on a thrshold of degree of the node
    def split_antennas_by_degree(self, threshold: int) -> list[set[int]]:
        """
        Splits the antennas into two sets based on a threshold of the degree of the node.
        Antennas with a degree less than or equal to the threshold will be grouped together.

        Args:
            threshold (int): The degree threshold for splitting antennas.

        Returns:
            list[set[int]]: A list of sets, where the first set contains antennas with a degree
            less than or equal to the threshold, and the second set contains the rest.
        """
        graph = self.loader.build_mis_instance_from_coordinates(self.antenna_range).graph
        low_degree_antennas = set()
        high_degree_antennas = set()

        for node in graph.nodes:
            if graph.degree(node) <= threshold:
                low_degree_antennas.add(node)
            else:
                high_degree_antennas.add(node)

        return [low_degree_antennas, high_degree_antennas]

    def reduce_colors(self) -> list[int]:
        """
        Attempts to reduce the number of colors used in the solution
        by trying to reassign every node of some color.
        Returns :
            list[int]: A list of colors for each antenna, where the index represents the antenna.
        """
        antennas = set([x for x in range(len(self.loader.coordinates_dataset))])
        graph = self.loader.build_mis_instance_from_coordinates(self.antenna_range, antennas).graph
        new_colors = deepcopy(self.colors)
        for color in range(self.colors_count):
            # Try to reassign all the nodes of the current color to a new color if possible
            possible_colors = set(range(self.colors_count)) - {color}
            assigned_all = True
            for node in range(len(self.colors)):
                possibilities = set(possible_colors)
                for neighbor in graph.neighbors(node):
                    if self.colors[neighbor] in possibilities:
                        possibilities.remove(self.colors[neighbor])
                if len(possibilities) > 0:
                    new_colors[node] = possibilities.pop()
                else:
                    assigned_all = False
            if assigned_all:
                # If we successfully reassigned all nodes of the current color
                self.colors = new_colors
                self.colors_count -= 1
                # Reassign the colors to be continuous
                for i in range(len(self.colors)):
                    if self.colors[i] > color:
                        self.colors[i] -= 1
        self.colors_count = max(self.colors) + 1
        return self.colors

    def check_solution(self) -> bool:
        """
        Check if the solution is valid by ensuring that no two antennas in the same color
        are within the antenna range of each other.

        Returns:
            bool: True if the solution is valid, False otherwise.
        """
        graph = self.loader.build_mis_instance_from_coordinates(self.antenna_range).graph
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                if self.colors[node] == self.colors[neighbor]:
                    return False
        return True

    def visualize_solution(self) -> plt:
        """
        Visualize the solution by plotting the antennas on a 2D plane.
        Each antenna is represented by a point, and antennas that are in the same
        independent set (i.e., do not interfere with each other) are colored the same.

        Returns:
            plt: A matplotlib plot object showing the antenna coverage solution.
        """
        plt.figure(figsize=(10, 8))
        for i, (lat, lon) in enumerate(self.loader.coordinates_dataset):
            plt.scatter(
                lon,
                lat,
                c=f"C{self.colors[i]}",
                label=f"Antenna {i}" if self.colors[i] == 0 else "",
                s=100,
            )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Antenna Coverage Solution")
        plt.grid()

        return plt
