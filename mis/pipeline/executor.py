from __future__ import annotations

from pulser import devices
from typing import Counter

from mis.pipeline.config import SolverConfig

from .execution import Execution
from .backends import QutipBackend
from .targets import Pulse, Register


class Executor:
    """
    Responsible for submitting compiled register and pulse to a backend.
    """

    def __init__(self, config: SolverConfig):
        """
        Args:
            config (SolverConfig): Solver configuration, including backend
                and device info.
            register (Register): The atom layout to execute.
            pulse (Pulse): The control signal to execute.
        """
        self.config = config

        device = config.device
        if device is None:
            device = devices.AnalogDevice

        backend = config.backend
        if backend is None:
            backend = QutipBackend(device=device)
        self.backend = backend

    def submit_job(self, pulse: Pulse, register: Register) -> Execution[Counter[str]]:
        """
        Submits the job to the backend and returns a processed MISSolution.

        Returns:
            The result of the execution.
        """
        return self.backend.run(register, pulse)
