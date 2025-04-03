from __future__ import annotations

from typing import Any

from mis.config import SolverConfig
from mis.types import BackendType

from .backends import QutipBackend, RemoteEmuMPSBackend, RemoteQPUBackend
from .targets import Pulse, Register


class Executor:
    """
    Responsible for submitting compiled register and pulse to a backend.
    """

    def __init__(self, config: SolverConfig):
        """
        Args:
            config (SolverConfig): Solver configuration, including backend and device info.
            register (Register): The atom layout to execute.
            pulse (Pulse): The control signal to execute.
        """
        self.config = config
        self.backend = self.get_backend()

    async def submit_job(self, pulse: Pulse, register: Register) -> Any:
        """
        Submits the job to the backend and returns a processed QUBOSolution.

        Returns:
            Any: The measured solution from execution.
        """
        self.register = register
        self.pulse = pulse
        result_counts = await self.backend.run(self.register, self.pulse)
        return result_counts

    def get_backend(self) -> Any:
        """
        Selects and instantiates the appropriate backend based on the config.

        Returns:
            An instantiated backend object ready for execution.
        """
        backend_type = BackendType(self.config.backend)

        if backend_type == BackendType.QUTIP:
            return QutipBackend(device=self.config.device)

        elif backend_type == BackendType.REMOTE_QPU:
            return RemoteQPUBackend(
                project_id=self.config.project_id,
                username=self.config.username,
                password=self.config.password,
                device_name=self.config.device.name,  # type: ignore[union-attr]
            )

        elif backend_type == BackendType.REMOTE_EMUMPS:
            return RemoteEmuMPSBackend(
                project_id=self.config.project_id,
                username=self.config.username,
                password=self.config.password,
                device_name=self.config.device.name,  # type: ignore[union-attr]
            )

        raise ValueError(f"Unsupported backend: {self.config.backend}")
