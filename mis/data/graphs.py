"""
Loading graphs as raw data.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from mis.pipeline.config import SolverConfig
from mis.shared.types import MISInstance
from mis.solver.solver import MISSolver


@dataclass
class DIMACSDataset:
    """
    A dataset representing a DIMACS graph instance and its solutions.
    This is used to load DIMACS files and extract the graph and solutions.
    """
    instance: MISInstance
    solutions: list[list[int]]


def load_dimacs(path: str) -> DIMACSDataset:
    """
    Load a DIMACS file and return a DIMACSDataset.

    Args:
        path (str): Path to the DIMACS file.

    Returns:
        DIMACSDataset: An instance of DIMACSDataset.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse the graph from the DIMACS format
    edges = []
    for line in lines:
        if line.startswith('p'):
            n_vertices, n_edges = map(int, line.split()[2:4])

        if line.startswith('e'):
            parts = line.split()
            edges.append((int(parts[1]), int(parts[2])))

    graph = nx.Graph()
    graph.add_edges_from(edges)

    if graph.number_of_nodes() != n_vertices:
        raise ValueError(
            "DIMACS file does not match specified number of vertices."
        )
    if graph.number_of_edges() != n_edges:
        raise ValueError(
            "DIMACS file does not match specified number of edges."
        )

    instance = MISInstance(graph)
    solver = MISSolver(instance, SolverConfig())
    solutions = [e.nodes for e in solver.solve().result()]
    return DIMACSDataset(instance=instance, solutions=solutions)
