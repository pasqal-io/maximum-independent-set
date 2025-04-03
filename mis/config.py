from __future__ import annotations

from dataclasses import dataclass

from pulser.devices import Device

from .types import BackendType

# Modules to be automatically added to the MISSolver namespace
__all__ = ["SolverConfig"]  # type: ignore


@dataclass
class SolverConfig:
    """
    Configuration class for setting up solver parameters.
    """

    use_quantum: bool = False
    """
    use_quantum (bool): When True, quantum solver is used to solve,
    otherwise classical methods are used.
    """
    backend: BackendType | None = None
    """
    backend (str): The name of the backend to use (e.g., 'qutip', 'remote_emumps').
    """
    max_iterations: int = 1
    """
    max_iterations (int): Maximum number of iterations allowed for solving.
    """
    device: Device | None = None
    """
    Pulser device to execute the code in
    """
    project_id: str = ""
    """
    Project ID on pasqal cloud
    """
    username: str = ""
    """
    Username on pasqal cloud
    """
    password: str = ""
    """
    Password on pasqal cloud
    """
