from __future__ import annotations

from pulser import devices
from typing import Any

from mis.config import SolverConfig
from mis.types import BackendType

from .backends import BaseBackend, QutipBackend, RemoteEmuMPSBackend, RemoteQPUBackend
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
        Submits the job to the backend and returns a processed MISSolution.

        Returns:
            Any: The measured solution from execution.
        """
        self.register = register
        self.pulse = pulse
        result_counts = await self.backend.run(self.register, self.pulse)
        return result_counts

    def get_backend(self) -> BaseBackend:
        """
        Selects and instantiates the appropriate backend based on the config.

        Returns:
            An instantiated backend object ready for execution.
        """
        backend_type = self.config.backend.backend

        if backend_type == BackendType.QUTIP:
            device = self.config.device
            if device is None:
                device = devices.AnalogDevice
            return QutipBackend(device=device)

        assert self.config.backend.project_id is not None
        assert self.config.backend.username is not None
        assert self.config.backend.password is not None
        device_name = None
        if self.config.device is not None:
            device_name = self.config.device.name
        if backend_type == BackendType.REMOTE_QPU:
            return RemoteQPUBackend(
                project_id=self.config.backend.project_id,
                username=self.config.backend.username,
                password=self.config.backend.password,
                device_name=device_name,
            )

        elif backend_type == BackendType.REMOTE_EMUMPS:
            return RemoteEmuMPSBackend(
                project_id=self.config.backend.project_id,
                username=self.config.backend.username,
                password=self.config.backend.password,
                device_name=device_name,
            )

        raise ValueError(f"Unsupported backend: {self.config.backend}")
