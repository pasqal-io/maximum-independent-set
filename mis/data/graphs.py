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
            continue  # Skip problem line
        if line.startswith('e'):
            parts = line.split()
            edges.append((int(parts[1]), int(parts[2])))

    graph = nx.Graph()
    graph.add_edges_from(edges)
    instance = MISInstance(graph)
    solver = MISSolver(instance, SolverConfig())
    solutions = [e.nodes for e in solver.solve().result()]
    return DIMACSDataset(instance=instance, solutions=solutions)
