from __future__ import annotations

from enum import Enum

class BackendType(str, Enum):
    """
    Type of backend to use for solving the QUBO
    """

    QUTIP = "qutip"
    REMOTE_QPU = "remote_qpu"
    REMOTE_EMUMPS = "remote_emumps"
