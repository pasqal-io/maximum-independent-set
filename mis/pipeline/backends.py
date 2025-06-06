"""
Executing a register and a sequence on a quantum device (including emulators).
"""

from __future__ import annotations

import abc
import os
from math import ceil
from time import sleep
from typing import Any, Counter, cast

from pasqal_cloud.batch import Batch
import pulser
from pasqal_cloud import SDK
from pasqal_cloud.device import BaseConfig, EmulatorType
from pulser import Sequence
from pulser.devices import Device
from pulser.json.abstract_repr.deserializer import deserialize_device
from pulser_simulation import QutipEmulator

import mis.pipeline.targets as targets
from mis.shared.error import CompilationError, ExecutionError
from .execution import Execution, Status, WaitingExecution


def make_sequence(
    device: Device, pulse: targets.Pulse, register: targets.Register
) -> pulser.Sequence:
    """
    Build a Pulser sequence for a device from a pulse and a register.

    This function is mostly intended for internal use and will likely move to qool-layer
    in time.

    Arguments:
        device: The quantum device for which the sequence is built. Used to detect if
            a pulse + register is not compatible with a device.
        pulse: The laser pulse to apply. It will be added as a Rydberg global channel.
        register: The geometry for the sequence. If the device expects an automatic
            layout, this must already have been normalized with `with_automatic_layout`.

    Raises:
        CompilationError if the pulse + register are not compatible with the device.
    """
    try:
        sequence = pulser.Sequence(register=register.register, device=device)
        sequence.declare_channel("ising", "rydberg_global")
        sequence.add(pulse.pulse, "ising")
        if pulse.detuning_maps is not None:
            for i, (map, wave) in enumerate(pulse.detuning_maps):
                dmm_id = f"dmm_{i}"
                sequence.config_detuning_map(map, dmm_id)
                sequence.add_dmm_detuning(wave, dmm_id)

        return sequence
    except ValueError as e:
        raise CompilationError(f"This pulse/register cannot be executed on the device: {e}")


class BaseBackend(abc.ABC):
    """
    Low-level abstraction to execute a Register and a Pulse on a Quantum Device.

    For higher-level abstractions, see `BaseExtractor` and its subclasses.

    The sole role of these abstractions is to provide the same API for all backends.
    They might be removed in a future version, once Pulser has gained a similar API.
    """

    def __init__(self, device: Device):
        self._device = device

    def _make_sequence(self, register: targets.Register, pulse: targets.Pulse) -> Sequence:
        assert self._device is not None
        return make_sequence(register=register, pulse=pulse, device=self._device)

    def device(self) -> Device:
        return self._device

    @abc.abstractmethod
    def run(self, register: targets.Register, pulse: targets.Pulse) -> Execution[Counter[str]]:
        raise NotImplementedError


class QutipBackend(BaseBackend):
    """
    Execute a Register and a Pulse on the Qutip Emulator.

    Please consider using EmuMPSBackend, which generally works much better with
    higher number of qubits.

    Performance warning:
        Executing anything quantum related on an emulator takes an amount of resources
        polynomial in 2^N, where N is the number of qubits. This can easily go beyond
        the limit of the computer on which you're executing it.
    """

    def __init__(self, device: Device | None = None):
        if device is None:
            device = pulser.devices.AnalogDevice
        super().__init__(device)

    def run(self, register: targets.Register, pulse: targets.Pulse) -> Execution[Counter[str]]:
        """
        Execute a register and a pulse.

        Arguments:
            register: The register (geometry) to execute. Typically obtained
                by compiling a graph.
            pulse: The pulse (lasers) to execute. Typically obtained by
                compiling a graph.

        Returns:
            A bitstring Counter, i.e. a data structure counting for each
            bitstring the number of instances of this bitstring observed
            at the end of runs.
        """
        sequence = self._make_sequence(register=register, pulse=pulse)
        emulator = QutipEmulator.from_sequence(sequence)
        result: Counter[str] = emulator.run().sample_final_state()
        return Execution.success(result)


class BaseRemoteExecution(WaitingExecution[Any]):
    """
    Execution on a remote device.

    Unless you're implementing a new backend, you
    probably want to use one of the subclasses.
    """

    def __init__(self, sleep_sec: int, batch: Batch):
        self._sleep_sec = sleep_sec
        self._batch = batch

    def status(self) -> Status:
        if self._batch.status in {"PENDING", "RUNNING"}:
            self._batch.refresh()
            return Status.IN_PROGRESS
        job = next(iter(self._batch.jobs.values()))
        if job.status == "ERROR":
            return Status.FAILURE
        return Status.SUCCESS

    def result(self) -> Any:
        while self.status() == Status.IN_PROGRESS:
            sleep(self._sleep_sec)
        job = next(iter(self._batch.jobs.values()))
        if self.status() == Status.FAILURE:
            raise ExecutionError(
                "Encountered errors while executing this " "sequence remotely: {}", job.errors
            )
        assert job.full_result is not None
        return job.full_result["counter"]


class BaseRemoteBackend(BaseBackend):
    """
    Base hierarch for remote backends.

    Performance warning:
        As of this writing, using remote Backends to access a remote QPU or
        remote emulator is slower than using a RemoteExtractor, as the
        RemoteExtractor optimizes the number of connections used to communicate
        with the cloud server.
    """

    def __init__(
        self,
        project_id: str,
        username: str,
        device_name: str | None = None,
        password: str | None = None,
    ):
        """
        Create a remote backend

        Args:
            project_id: The ID of the project on the Pasqal Cloud API.
            username: Your username on the Pasqal Cloud API.
            password: Your password on the Pasqal Cloud API. If you leave
                this to None, you will need to enter your password manually.
            device_name: The name of the device to use. As of this writing,
                the default value of "FRESNEL" represents the latest QPU
                available through the Pasqal Cloud API.
        """
        if device_name is None:
            device_name = "FRESNEL"
        self.device_name = device_name
        self._sdk = SDK(username=username, project_id=project_id, password=password)
        self._max_runs = 500
        self._sequence = None
        self._device = None
        super().__init__(device=self._fetch_device())  # FIXME: Currently sync.

    def _fetch_device(self) -> Device:
        """
        Make sure that we have fetched the latest specs for the device from
        the server.
        """
        # FIXME: With a remote backend, truly, this should be async.
        if self._device is not None:
            return self._device

        # Fetch the latest list of QPUs
        # Implementation note: Currently sync, hopefully async in the future.
        specs = self._sdk.get_device_specs_dict()
        self._device = cast(Device, deserialize_device(specs[self.device_name]))

        # As of this writing, the API doesn't support runs longer than
        # 500 jobs. If we want to add more runs, we'll need to split them
        # across several jobs.
        if isinstance(self._device.max_runs, int):
            self._max_runs = self._device.max_runs

        return self._device

    def _extract(self, payload: Counter) -> Counter[str]:
        # We expect that the payload returned will always be a `Counter[str]`,
        # but we still need to double-check.
        assert isinstance(payload, Counter)
        if len(payload) == 0:
            return payload
        k, v = next(iter(payload))
        assert isinstance(k, str)
        assert isinstance(v, int)
        return payload

    def _run(
        self,
        register: targets.Register,
        pulse: targets.Pulse,
        emulator: EmulatorType | None,
        config: BaseConfig | None = None,
        sleep_sec: int = 2,
    ) -> Execution[Counter[str]]:
        """
        Run the pulse + register.

        Arguments:
            register: A register to run.
            pulse: A pulse to execute.
            emulator: The emulator to use, or None to run on a QPU.
            config: The backend-specific config.
            sleep_sec (optional): The amount of time to sleep when waiting for
                the remote server to respond, in seconds. Defaults to 2.

        Raises:
            CompilationError: If the register/pulse may not be executed on
                this device.
        """
        device = self._fetch_device()
        try:
            sequence = make_sequence(device=device, pulse=pulse, register=register)

            self._sequence = sequence
        except ValueError as e:
            raise CompilationError("This register/pulse cannot be executed " f"on the device: {e}")

        # Enqueue execution.
        batch = self._sdk.create_batch(
            serialized_sequence=sequence.to_abstract_repr(),
            jobs=[{"runs": self._max_runs}],
            wait=False,
            emulator=emulator,
            configuration=config,
        )

        return BaseRemoteExecution(sleep_sec=sleep_sec, batch=batch).map(self._extract)


class RemoteQPUBackend(BaseRemoteBackend):
    """
    Execute on a remote QPU.

    Performance note:
        As of this writing, the waiting lines for a QPU
        may be very long. You may use this Extractor to resume your workflow
        with a computation that has been previously started.
    """

    def run(self, register: targets.Register, pulse: targets.Pulse) -> Execution[Counter[str]]:
        return self._run(register, pulse, emulator=None, config=None)


class RemoteEmuMPSBackend(BaseRemoteBackend):
    """
    A backend that uses a remote high-performance emulator (EmuMPS)
    published on Pasqal Cloud.
    """

    def _extract(self, payload: Any) -> Counter[str]:
        return super()._extract(payload)

    def run(
        self, register: targets.Register, pulse: targets.Pulse, dt: int = 10
    ) -> Execution[Counter[str]]:
        return self._run(register, pulse, emulator=None, config=None)


if os.name == "posix":
    import emu_mps

    class EmuMPSBackend(BaseBackend):
        """
        Execute a Register and a Pulse on the high-performance emu-mps
        Emulator.

        As of this writing, this local emulator is only available under Unix.
        However, the RemoteEmuMPSBackend is available on all platforms.

        Performance warning:
            Executing anything quantum related on an emulator takes an amount
            of resources polynomial in 2^N, where N is the number of qubits.
            This can easily go beyond the limit of the computer on which
            you're executing it.
        """

        def __init__(self, device: Device):
            super().__init__(device)

        def run(
            self, register: targets.Register, pulse: targets.Pulse, dt: int = 10
        ) -> Execution[Counter[str]]:
            sequence = self._make_sequence(register=register, pulse=pulse)
            backend = emu_mps.MPSBackend()

            # Configure observable.
            cutoff_duration = int(ceil(sequence.get_duration() / dt) * dt)
            observable = emu_mps.BitStrings(evaluation_times={cutoff_duration})
            config = emu_mps.MPSConfig(observables=[observable], dt=dt)
            counter: Counter[str] = backend.run(sequence, config)[observable.name][cutoff_duration]
            return Execution.success(counter)
