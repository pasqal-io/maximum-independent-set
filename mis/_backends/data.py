from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Counter

import pulser
import pydantic
from pulser.backend.remote import BatchStatus
from pulser.register import QubitId
from pulser.waveforms import Waveform

from mis._backends.types import BackendType, DeviceType

logger = logging.getLogger(__name__)


@dataclass
class Detuning:
    """A single detuning channel."""

    weights: dict[QubitId, float]
    """Association of weights to qubits."""

    waveform: Waveform
    """The waveform for this detuning channel."""


@dataclass
class QuantumProgram:
    """
    The program to execute on the QPU/emulator.

    In time, this class is expected to disappear and be replaced with `qoolqit.QuantumProgram`.
    """

    device: pulser.devices.Device
    register: pulser.Register
    pulse: pulser.Pulse
    detunings: list[Detuning] = field(default_factory=list)
    """
    Detunings.

    As of this writing, they're not supported in qoolqit.QuantumProgram.
    """


pydantic.BaseModel.model_config["arbitrary_types_allowed"] = True


class NamedDevice(str):
    """An individual, named, device, e.g. "FRESNEL"."""

    pass


class BackendConfig(pydantic.BaseModel):
    """Generic configuration for backends."""

    backend: BackendType = BackendType.QUTIP
    """
    The type of backend to use.

    If None, pick a reasonable default backend running locally.
    """

    username: str | None = None
    """
    For a backend that requires authentication, such as Pasqal Cloud,.

    the username.

    Otherwise, ignored.
    """

    password: str | None = None
    """
    For a backend that requires authentication, such as Pasqal Cloud,.

    the password.

    Otherwise, ignored.
    """

    project_id: str | None = None
    """
    For a backend that associates jobs to projects, such as Pasqal Cloud,.

    the id of the project. The project must already exist.

    Otherwise, ignored.
    """

    device: (
        NamedDevice | DeviceType | pulser.devices.Device | pulser.devices.VirtualDevice | None
    ) = None
    """
    For a backend that supports numerous devices, either:

    - a type of device (e.g. `DeviceType.ANALOG_DEVICE`); or
    - the name of a specific device (e.g. `NamedDevice("FRESNEL")`).
    - a pulser Device or VirtualDevice.

    If unspecified, pick a backend-appropriate device.
    """

    dt: int = 10
    """
    For a backend that supports customizing the duration of steps, the.

    timestep size.

    As of this writing, this parameter is used only by the EmuMPS backends.
    """

    wait: bool = False
    """
    For a remote backend where we submit a batch of jobs,.

    block execution on this statement until all the submitted jobs are terminated .
    """


class CompilationError(Exception):
    """
    An error raised when attempting to compile a graph for an architecture.

    that does not support it, e.g. because it requires too many qubits or
    because the physical constraints on the geometry are not satisfied.
    """

    pass


class ExecutionError(Exception):
    """An error during the execution of a job."""

    pass


class JobId(str):
    """A unique identifier for a job."""

    pass


@dataclass
class Result:
    """
    Low-level results returned from a backend.

    Specific backends may return subclasses of this class with additional
    backend-specific information.
    """

    counts: Counter[str]
    """
    A mapping from bitstrings observed to the number of instances of this.

    bitstring observed.
    """

    def __len__(self) -> int:
        """The total number of measures."""
        return sum(self.counts.values())


@dataclass
class BaseJob(ABC):
    """
    A job, either pending or in progress.

    To wait until the job is complete, use `await job`.
    """

    id: JobId
    """
    The unique identifier for this job.

    You may save it to a database and use `Backend.proceed` to recreate
    a job from a JobId.
    """

    @abstractmethod
    def wait(self) -> Result:
        """
        Wait until the job is complete, blocking the entire thread until it is.

        Once the job is complete (or if it is already complete), return
        the result of the job. If the job failed, raise an ExecutionError.

        # Performance note

        This method is expected to spend most of its time outside the GIL, which
        means that if you run it on a background thread, it should not impact the
        performance of other threads.
        """
        ...

    @abstractmethod
    def status(self) -> BatchStatus:
        """
        Check the status of the job.

        This method is provided as a polling mechanism, mainly to help writing client
        code in libraries or applications that need to wait for the completion of numerous
        concurrent jobs. If you are simply interested in the result of a single job,
        you should rather use method `wait()`.
        """
        ...


@dataclass
class JobSuccess(BaseJob):
    """A job that has already succeeded."""

    result: Result

    def status(self) -> BatchStatus:
        return BatchStatus.DONE

    def wait(self) -> Result:
        return self.result


@dataclass
class JobFailure(BaseJob):
    """A job that has already failed."""

    error: Exception

    def status(self) -> BatchStatus:
        return BatchStatus.ERROR

    def wait(self) -> Result:
        raise self.error
