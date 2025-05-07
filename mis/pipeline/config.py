from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pulser.devices import Device

if TYPE_CHECKING:
    from mis.pipeline.backends import BaseBackend
    from mis.pipeline.embedder import BaseEmbedder
    from mis.pipeline.pulse import BasePulseShaper
    from mis.pipeline.preprocessor import BasePreprocessor
from mis.shared.types import MethodType

# Modules to be automatically added to the MISSolver namespace
__all__ = ["SolverConfig"]  # type: ignore


@dataclass
class SolverConfig:
    """
    Configuration class for setting up solver parameters.
    """

    backend: BaseBackend | None = None
    """
    backend (optional): Backend configuration to use. If `None`,
    use a reasonable default emulator.
    """

    method: MethodType = MethodType.EAGER
    """
    method: The method used to solve this instance of MIS.
    If unspecified, use classical (non-quantum) MIS.
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

    preprocessor: BasePreprocessor | None = None
    """
    preprocessor: If specified, a graph preprocessor, used to decrease
        the size of the graph (hence the duration of actual resolution)
        by applying heuristics prior to embedding on a quantum device.

        By default, apply Kernelization, a set of non-destructive operations
        that reduce the size of the graph prior to solving the problem.
        This preprocessor reduces the number of qubits needed to execute
        the embedded graph on the quantum device.

        If you wish to deactivate preprocessing entirely, pass the
        `EmptyPreprocessor()`.

        If you wish to apply more than one preprocessor, you will
        need to specify in which order these preprocessurs must be called,
        or if some of them need to be called more than once, etc. For
        this purpose, you'll need to write your own subclass of
        `BasePreprocessor` that orchestrates calling the individual
        preprocessors.
    """
