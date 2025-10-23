from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import pulser
from pulser import Sequence
from pulser.devices import Device, VirtualDevice

from mis._backends.data import BackendConfig, BaseJob, JobId, QuantumProgram, Result
from qoolqit.exceptions import CompilationError

logger = logging.getLogger(__name__)


def make_sequence(program: QuantumProgram) -> pulser.Sequence:
    """
    Build a sequence for a device from a pulse and a register.

    This function is mostly intended for internal use and will likely move to qool-layer
    in time.

    Arguments:
        program: A quantum program to compile into a sequence.

    Raises:
        CompilationError if the pulse + register are not compatible with the device.
    """
    # Normalize and apply register.
    register = deepcopy(program.register)
    if (
        isinstance(program.device, Device)
        and program.device.requires_layout
        and register.layout is None
    ):
        register = program.register.with_automatic_layout(program.device)
    sequence = Sequence(register=register, device=program.device)

    # Add global pulse.
    sequence.declare_channel("rydberg_global", "rydberg_global")
    sequence.add(program.pulse, "rydberg_global")

    # Add any detuning pulse.
    if len(program.detunings) > 0:
        channels = list(program.device.dmm_channels.keys())
        if len(channels) == 0:
            raise CompilationError(
                f"This program specifies {len(program.detunings)} detunings but "
                "the device doesn't offer any DMM channel to execute them."
            )
        # Arbitrarily pick the first channel.
        dmm_id = channels[0]
        for detuning in program.detunings:
            detuning_map = register.define_detuning_map(detuning_weights=detuning.weights)
            sequence.config_detuning_map(detuning_map, dmm_id=dmm_id)
            sequence.add_dmm_detuning(detuning.waveform, dmm_id)

    return sequence


class BaseBackend(ABC):
    """
    Base class for backends.

    # Implementation note

    If you implement a new backend, please:

    1. Make sure that `__init__` takes exactly the same arguments as
        `BaseBackend.__init__`.
    2. Register it as part of `BackendType`
    2. Make sure that it can be executed on a background thread.
    """

    def __init__(self, config: BackendConfig):
        self.config = config

    def default_number_of_runs(self) -> int:
        """A backend-specific reasonable default value for the number of runs."""
        # Reasonable default.
        return 100

    @abstractmethod
    def device(self) -> Device | VirtualDevice:
        """
        Specifications for the device picked by `BackendConfig.device`.

        If
        no such device was specified, return the default device for this
        backend.

        Note that any client (caller, etc.) MUST call `device()` to find out
        about the specific device, rather than instantiating a device directly
        from `pulser`. If your client ever calls a remote QPU, this is the
        ONLY way of being certain that you have access to the latest version
        of the QPU specs.

        # Performance note

        This method is expected to spend most of its time outside the GIL, which
        means that if you run it on a background thread, it should not impact the
        performance of other threads.
        """
        ...

    def run(self, program: QuantumProgram, runs: int | None = None) -> Result:
        """
        Submit a quantum program for execution and wait for its result.

        Arguments:
            runs: How many times the program must be executed on the backend.
                If `None`, pick a reasonable default.

        Note that if you are submitting a long job and expecting the need
        to resume it later, you should rather use `submit` and `proceed`.

        # Performance note

        This method is expected to spend most of its time outside the GIL, which
        means that if you run it on a background thread, it should not impact the
        performance of other threads.
        """
        return self.submit(program, runs).wait()

    @abstractmethod
    def submit(self, program: QuantumProgram, runs: int | None = None) -> BaseJob:
        """
        Submit a quantum program for execution.

        Arguments:
            runs: How many times the program must be executed on the backend.
                If `None`, pick a reasonable default.

                        Once a program is submitted, you can obtain a `BaseJob` from its job id
        by calling `proceed()`. This can be useful if you enqueue a program on a
        long queue (e.g. one in which you may need to wait for hours or days before
        you have access to a QPU), save the job id, turn off your computer, then
        resume your session a few days later to check the status of the program.

        # Performance note

        This method is expected to spend most of its time outside the GIL, which
        means that if you run it on a background thread, it should not impact the
        performance of other threads.
        """
        ...

    @abstractmethod
    def proceed(self, job: JobId) -> BaseJob:
        """
        Continue tracking execution of a quantum program submitted previously.

        This may be useful, for instance, if you have launched a remote quantum
        program during a previous session and now wish to check its result

        # Performance note

        This method is expected to spend most of its time outside the GIL, which
        means that if you run it on a background thread, it should not impact the
        performance of other threads.
        """
        ...
