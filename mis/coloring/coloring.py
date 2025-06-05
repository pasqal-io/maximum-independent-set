from mis import MISSolver
from mis.data.dataloader import DataLoader
from mis.pipeline.basesolver import BaseSolver
from mis.pipeline.config import SolverConfig
import matplotlib.pyplot as plt

from mis.pipeline.execution import Execution
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

    def solve(self) -> Execution[list[MISSolution]]:
        """
        Solve the graph coloring problem by finding a maximum independent set
        for the given antenna range and coloring the antennas accordingly.
        Returns:
            Execution[list[MISSolution]]: An execution object containing the nodes of each color in the solution.
        """
        antennas = set([x for x in range(len(self.loader.coordinates_dataset))])
        self.colors = [-1] * len(antennas)

        res = []
        self.colors_count = 0
        while len(antennas) > 0:
            solver = MISSolver(
                self.loader.build_mis_instance_from_coordinates(self.antenna_range, antennas),
                self.solver_config,
            )
            solutions = solver.solve().result()
            res.append(solutions[0])
            for antenna in solutions[0].nodes:
                self.colors[antenna] = self.colors_count
            antennas = antennas - set(solutions[0].nodes)
            self.colors_count += 1

        return Execution.success(res)

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
