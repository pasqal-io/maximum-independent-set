from __future__ import annotations

from data import MISInstance, MISSolution
from solver import MISSolver

"""
The maximum independent set is a Python library designed for the machine learning community to help users design quantum-driven similarity metrics for graphs and to use them inside kernel-based machine learning algorithms for graph data.

The core of the library is focused on the development of a classification algorithm for molecular-graph dataset as it is presented in the published paper [Quantum feature maps for graph machine learning on a neutral atom quantum processor](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042615).

Users setting their first steps into quantum computing will learn how to implement the core algorithm in a few simple steps and run it using the Pasqal Neutral Atom QPU. More experienced users will find this library to provide the right environment to explore new ideas - both in terms of methodologies and data domain - while always interacting with a simple and intuitive QPU interface.
"""

# list_of_submodules = [
#     "data",
# ]

# __all__ = []
# for submodule in list_of_submodules:
#     __all_submodule__ = getattr(import_module(submodule, package="mis"), "__all__")
#     __all__ += __all_submodule__


__all__ = ["MISSolver", "MISInstance", "MISSolution"]
