from __future__ import annotations

from enum import Enum

from pulser.devices import AnalogDevice, DigitalAnalogDevice


class StrEnum(str, Enum):
    """String-based Enums class implementation."""

    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def names(cls) -> list[str]:
        return list(map(lambda c: c.name, cls))  # type: ignore

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class BackendType(StrEnum):
    """Type of backend to use for solving the QUBO."""

    QUTIP = "qutip"
    REMOTE_QPU = "remote_qpu"
    REMOTE_EMUMPS = "remote_emumps"
    REMOTE_EMUTN = "remote_emutn"
    REMOTE_EMUFREE = "remote_emufree"
    EMU_MPS = "emu_mps"
    EMU_SV = "emu_sv"


class DeviceType(Enum):
    """Type of device to use for the solver."""

    ANALOG_DEVICE = AnalogDevice
    DIGITAL_ANALOG_DEVICE = DigitalAnalogDevice
