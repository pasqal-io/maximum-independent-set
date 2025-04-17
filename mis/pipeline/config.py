from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pulser.devices import Device

if TYPE_CHECKING:
    from mis.pipeline.backends import BaseBackend
    from mis.pipeline.embedder import BaseEmbedder
    from mis.pipeline.pulse import BasePulseShaper
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

    Only needed if `use_quantum` is `True`.
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

    device: Device | None = None
    """
    Quantum device to execute the code in. If unspecified, use a
    reasonable default device.
    """

    embedder: BaseEmbedder | None = None
    pulse_shaper: BasePulseShaper | None = None
