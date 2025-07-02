"""
Configuration for MIS solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import networkx as nx

from pulser.devices import Device

if TYPE_CHECKING:
    from mis.pipeline.embedder import BaseEmbedder
    from mis.pipeline.pulse import BasePulseShaper
    from mis.pipeline.postprocessor import BasePostprocessor
    from mis.pipeline.preprocessor import BasePreprocessor
from mis.shared.types import MethodType

from qoolqit._solvers import BackendType  # noqa: F401 # imported for re-export
from qoolqit._solvers.backends import BaseBackend, BackendConfig

# Modules to be automatically added to the MISSolver namespace
__all__ = [
    "BackendConfig",
    "BackendType",
    "SolverConfig",
]


def default_preprocessor() -> Callable[[nx.Graph], BasePreprocessor]:
    """
    Instantiate the default preprocessor.

    As of this writing, the default preprocessor is mis.pipeline.kernelization.Kernelization.
    """
    # Avoid circular dependencies during load.
    from mis.pipeline.kernelization import Kernelization

    return lambda graph: Kernelization(graph)


def default_postprocessor() -> BasePostprocessor:
    """
    Instantiate the default postprocessor.

    As of this writing, the default postprocessor is mis.pipeline.maximization.Maximization.
    """
    # Avoid circular dependencies during load.
    from mis.pipeline.maximization import Maximization

    return Maximization()


@dataclass
class GreedyConfig:
    """
    Configuration for greedy solving strategies.
    """

    default_solving_threshold: int = 2
    """
    default_solving_threshold (int): Size threshold (number of nodes) for using MIS solving
    when greedy method is used.

    If a subgraph has a number of nodes less than or equal to this value, it will be solved
    using the default solver.
    """

    subgraph_quantity: int = 5
    """
    subgraph_quantity (int): Number of candidate subgraphs to generate during greedy mapping.

    This defines how many alternative graph-to-layout mappings will be generated and evaluated.
    Increasing this may improve solution quality but also increases runtime.
    """

    mis_sample_quantity: int = 1
    """
    mis_sample_quantity (int): Number of MIS solutions to sample per iteration (if applicable).
    """


@dataclass
class SolverConfig:
    """
    Configuration class for setting up solver parameters.
    """

    backend: BaseBackend | BackendConfig | None = None
    """
    backend (optional): Backend configuration to use. If `None`,
    use a non-quantum heuristic solver.
    """

    method: MethodType = MethodType.EAGER
    """
    method: The method used to solve this instance of MIS.
    """

    max_iterations: int = 1
    """
    max_iterations (int): Maximum number of iterations allowed for solving.
    """

    max_number_of_solutions: int = 1
    """
    A maximal number of solutions to return.

    The solver will return up to `max_number_of_solutions` solutions, ranked
    from most likely to least likely. Some solvers will only return a single
    solution.
    """

    device: Device | None = None
    """
    Quantum device to execute the code in. If unspecified, use a
    reasonable default device.
    """

    embedder: BaseEmbedder | None = None
    """
    embedder: If specified, an embedder, i.e. a mechanism used
        to customize the layout of neutral atoms on the quantum
        device. Ignored for non-quantum backends.
    """

    pulse_shaper: BasePulseShaper | None = None
    """
    pulse_shaper: If specified, a pulse shaper, i.e. a mechanism used
        to customize the laser pulse to which the neutral atoms are
        subjected during the execution of the quantum algorithm.
        Ignored for non-quantum backends.
    """

    preprocessor: Callable[[nx.Graph], BasePreprocessor] | None = field(
        default_factory=default_preprocessor
    )
    """
    preprocessor: A graph preprocessor, used to decrease
        the size of the graph (hence the duration of actual resolution)
        by applying heuristics prior to embedding on a quantum device.

        By default, apply Kernelization, a set of non-destructive operations
        that reduce the size of the graph prior to solving the problem.
        This preprocessor reduces the number of qubits needed to execute
        the embedded graph on the quantum device.

        If you wish to deactivate preprocessing entirely, pass `None`.

        If you wish to apply more than one preprocessor, you will
        need to specify in which order these preprocessurs must be called,
        or if some of them need to be called more than once, etc. For
        this purpose, you'll need to write your own subclass of
        `BasePreprocessor` that orchestrates calling the individual
        preprocessors.
    """

    postprocessor: Callable[[], BasePostprocessor] | None = default_postprocessor
    """
        A postprocessor used to sort out and improve results.

        By default, apply Maximization, a set of heuristics that attempt
        to "fix" quantum results in case of accidental bitflips.

        If you wish to deactivate postprocessing entirely, pass `None`.
    """

    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    """
    If specified, use this for solving the GreedyMIS.
    Needs to be specified when method is GreedyMIS
    """

    runs: int = 500
    """
    When using a quantum backend, how many times we repeat each quantum run.
    Since quantum executions are non-deterministic, higher numbers increases
    the reliability of the result, but also the duration until the results are
    available.
    """
