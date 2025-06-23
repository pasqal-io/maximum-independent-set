"""
Loading graphs as raw data.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from mis.shared.types import MISInstance
from pathlib import Path


@dataclass
class DIMACSDataset:
    """
    A dataset representing a DIMACS graph instance and its solutions.
    This is used to load DIMACS files and extract the graph and solutions.
    """

    instance: MISInstance
    solutions: list[list[int]]
    """
    If the dataset provided solutions, the list of solutions.
    Note that most instances do not provide a solution and those that do do not guarantee that they provide all solutions.
    """


def load_dimacs(path: Path) -> DIMACSDataset:
    """
    Load a DIMACS file and return a DIMACSDataset.

    Args:
        path (Path): Path to the DIMACS file.

    Returns:
        DIMACSDataset: An instance of DIMACSDataset.
    """
    lines = path.read_text().splitlines()

    # Parse the graph from the DIMACS format
    edges = []
    solutions = []
    n_vertices = None
    n_edges = None
    for line in lines:
        if line.startswith("p"):
            if n_vertices is not None:
                raise ValueError("Error in DIMACS file: duplicate number of vertices/edges.")
            n_vertices, n_edges = map(int, line.split()[2:4])

        if line.startswith("e"):
            parts = line.split()
            edges.append((int(parts[1]), int(parts[2])))

        if line.startswith("max indep set"):
            _, line = line.split("=", 1)
            solutions = [list(map(int, line.strip(" {}").split(",")))]
    if n_vertices is None:
        raise ValueError("Error in DIMACS file: missing number of vertices.")
    if n_edges is None:
        raise ValueError("Error in DIMACS file: missing number of edges.")

    graph = nx.Graph()
    graph.add_nodes_from(range(1, n_vertices + 1))
    graph.add_edges_from(edges)

    if graph.number_of_nodes() != n_vertices:
        raise ValueError(
            "Error in DIMACS file: the graph within this file does not have the number of vertices it claims."
        )
    if graph.number_of_edges() != n_edges:
        raise ValueError(
            "Error in DIMACS file: the graph within this file does not have the number of edges it claims."
        )

    instance = MISInstance(graph)
    return DIMACSDataset(instance=instance, solutions=solutions)
