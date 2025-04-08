from __future__ import annotations

from dataclasses import dataclass, field

from pulser.devices import Device

from .types import BackendType, MethodType

# Modules to be automatically added to the MISSolver namespace
__all__ = ["SolverConfig"]  # type: ignore


@dataclass
class BackendConfig:
    @staticmethod
    def default() -> BackendConfig:
        """
        A default backend to use.

        This backend will be an emulator running on your computer.
        """
        return BackendConfig(backend=BackendType.QUTIP)

    @staticmethod
    def remote_qpu(project_id: str, username: str, password: str) -> BackendConfig:
        """
        Use a QPU on the Pasqal Cloud
        """
        return BackendConfig(
            backend=BackendType.REMOTE_QPU,
            project_id=project_id,
            username=username,
            password=password,
        )

    backend: BackendType
    """
    backend (str): The name of the backend to use (e.g., 'qutip', 'remote_emumps').
    """

    project_id: str | None = None
    """
    Project ID on pasqal cloud.

    Only needed if you are using a cloud backend.
    """

    username: str | None = None
    """
    Username on pasqal cloud

    Only needed if you are using a cloud backend.
    """

    password: str | None = None
    """
    Password on pasqal cloud

    Only needed if you are using a cloud backend.
    """


@dataclass
class SolverConfig:
    """
    Configuration class for setting up solver parameters.
    """

    method: MethodType = MethodType.CLASSICAL
    """
    method: The method used to solve this instance of MIS.
    If unspecified, use classical (non-quantum) MIS.
    """

    backend: BackendConfig = field(default_factory=BackendConfig.default)
    """
    backend (optional): Backend configuration to use. If `None`,
    use a reasonable default emulator.

    Only needed if `use_quantum` is `True`.
    """

    max_iterations: int = 1
    """
    max_iterations (int): Maximum number of iterations allowed for solving.
    """

    device: Device | None = None
    """
    Pulser device to execute the code in
    """
