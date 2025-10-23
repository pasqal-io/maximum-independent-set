from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Counter
from uuid import uuid4

import emu_mps
import emu_sv
import pulser
from pulser import Sequence
from pulser.devices import Device, VirtualDevice
from pulser_simulation import QutipEmulator

from mis._backends.data import (
    BackendConfig,
    BaseJob,
    JobFailure,
    JobId,
    JobSuccess,
    NamedDevice,
    QuantumProgram,
    Result,
)
from mis._backends.types import DeviceType

from .base_backend import BaseBackend, make_sequence

logger = logging.getLogger(__name__)


############################ Local backends


class BaseLocalBackend(BaseBackend):
    """
    Base class for emulators running locally.

    To implement a new local backend, you only need to provide an implementation
    of method `_execute_locally`.
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        device = config.device
        if device is None:
            # Note that the choice of this device is somewhat arbitrary.
            # We pick DigitalAnalogDevice because it supports DMM and we
            # need DMM for some algorithms, but in the future, we might
            # change this.
            device = pulser.DigitalAnalogDevice
        elif isinstance(device, NamedDevice):
            raise ValueError(
                "Local emulators do not support named devices, property `device` expects `None` "
                + "or a `DeviceType`"
            )
        elif isinstance(device, DeviceType):
            device = device.value
        assert isinstance(device, (Device, VirtualDevice)), f"Expected a device, got {device}"
        self._device = device

    def device(self) -> Device:
        return self._device

    def submit(self, program: QuantumProgram, runs: int | None = None) -> BaseJob:
        id = JobId(str(object=uuid4()))
        sequence = make_sequence(program)
        try:
            result = self._execute_locally(sequence, runs)
            return JobSuccess(id=id, result=result)
        except Exception as e:
            return JobFailure(id=id, error=e)

    @abstractmethod
    def _execute_locally(self, sequence: Sequence, runs: int | None = None) -> Result:
        """
        Execute a quantum program locally.

        Arguments:
            sequence: The Pulser sequence to execute.
            runs: The number of runs for the execution. If `None`, the backend should
                default to a reasonable number of runs.
        """
        ...

    def proceed(self, job: JobId) -> BaseJob:
        # FIXME: Implement save/restore job results.
        raise NotImplementedError()


class QutipBackend(BaseLocalBackend):
    """
    Execute a Register and a Pulse on the Qutip Emulator.

    Please consider using EmuMPSBackend, which generally works much better with
    higher number of qubits.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, config: BackendConfig):
        super().__init__(config)

    def _execute_locally(self, sequence: Sequence, runs: int | None = None) -> Result:
        emulator = QutipEmulator.from_sequence(sequence)
        if runs is None:
            runs = 100  # Arbitrary device-specific value.
        result: Counter[str] = emulator.run().sample_final_state(N_samples=runs)
        return Result(counts=result)


class EmuMPSBackend(BaseLocalBackend):
    """
    Execute a Register and a Pulse on the high-performance emu-mps Emulator.

    As of this writing, this local emulator is only available under Unix. However,
    the RemoteEmuMPSBackend is available on all platforms.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

    def _execute_locally(self, sequence: Sequence, runs: int | None = None) -> Result:
        times = [1.0]  # 1.0 = end of the duration (normalized)
        if runs is None:
            runs = 100  # Arbitrary device-specific value.
        bitstrings = emu_mps.BitStrings(evaluation_times=times, num_shots=runs)
        config = emu_mps.MPSConfig(observables=[bitstrings], dt=self.config.dt)
        backend = emu_mps.MPSBackend(sequence, config=config)
        results = backend.run()
        counter: Counter[str] = results.bitstrings[-1]
        return Result(counts=counter)


class EmuSVBackend(BaseLocalBackend):
    """
    Execute a Register and a Pulse on the high-performance emu-sv Emulator.

    As of this writing, this local emulator is only available under Unix.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)

    def _execute_locally(self, sequence: Sequence, runs: int | None = None) -> Result:
        times = [1.0]  # 1.0 = end of the duration (normalized)
        if runs is None:
            runs = 100  # Arbitrary device-specific value.
        bitstrings = emu_sv.BitStrings(evaluation_times=times, num_shots=runs)
        config = emu_sv.SVConfig(dt=self.config.dt, observables=[bitstrings], log_level=0)
        backend = emu_sv.SVBackend(sequence, config=config)

        results = backend.run()
        counter: Counter[str] = results.get_result(bitstrings, time=1.0)
        return Result(counts=counter)
