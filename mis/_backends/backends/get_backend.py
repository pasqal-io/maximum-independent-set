from __future__ import annotations

from typing import cast

from mis._backends.types import BackendType

from .base_backend import BackendConfig, BaseBackend
from .local_backends import EmuMPSBackend, EmuSVBackend, QutipBackend
from .remote_backends import (
    RemoteEmuFREEBackend,
    RemoteEmuMPSBackend,
    RemoteQPUBackend,
)

backends_map: dict[BackendType, type[BaseBackend]] = {
    BackendType.QUTIP: cast(type[BaseBackend], QutipBackend),
    BackendType.EMU_MPS: cast(type[BaseBackend], EmuMPSBackend),
    BackendType.EMU_SV: cast(type[BaseBackend], EmuSVBackend),
    BackendType.REMOTE_EMUMPS: cast(type[BaseBackend], RemoteEmuMPSBackend),
    BackendType.REMOTE_QPU: cast(type[BaseBackend], RemoteQPUBackend),
    BackendType.REMOTE_EMUFREE: cast(type[BaseBackend], RemoteEmuFREEBackend),
}


def get_backend(backend_config: BackendConfig) -> BaseBackend:
    """
    Instantiate a backend.

    # Concurrency note
    Backends are *not* meant to be shared across threads.
    """
    backend = backends_map.get(backend_config.backend, None)
    if backend is not None:
        return backend(backend_config)
    else:
        raise ValueError(f"Unknown backend {backend_config.backend}.")
