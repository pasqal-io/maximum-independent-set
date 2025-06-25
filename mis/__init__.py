"""
Maximum Independent Set is a Python library that provides quantum solvers for the Maximum Independent Set problem.

The Maximum Independent Set problem (or MIS) is a graph optimization problem with applications to numerous fields, e.g.
scheduling, constraint solving, etc. The input of this problem is a set of nodes and a set of conflicts between these
nodes. Solving MIS is finding the largest set of nodes that do not have conflicts. This problem is well-known for being
NP-hard -- in other words, there are no known exact, non-exponential algorithms that can reliably solve the problem,
forcing entire industries to rely upon heuristics. Quantum computers offer alternative approaches to solving a number of
problems, including MIS, in reasonable time.

The core of the library is a set of MIS solvers developed to be used on quantum devices, including quantum computers, for
users who have access to one, or quantum emulators, for users who do not. We also provide non-quantum MIS solvers for
benchmarking purposes.
"""

from __future__ import annotations

from qoolqit._solvers.backends import BackendConfig, BackendType
from .solver.solver import MISSolver
from .pipeline.config import SolverConfig
from .shared.types import MISInstance

__all__ = [
    "MISSolver",
    "MISInstance",
    "SolverConfig",
    "BackendConfig",
    "BackendType",
]
