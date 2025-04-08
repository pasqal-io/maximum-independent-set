from __future__ import annotations

from enum import Enum


class BackendType(str, Enum):
    """
    Type of backend to use for solving the MIS
    """

    QUTIP = "qutip"
    REMOTE_QPU = "remote_qpu"
    REMOTE_EMUMPS = "remote_emumps"


class MethodType(str, Enum):
    CLASSICAL = "classical"
    CLASSICAL_GREEDY = "classical-greedy"
    QUANTUM = "quantum"
    QUANTUM_GREEDY = "quantum-greedy"

    def is_quantum(self) -> bool:
        return self in [MethodType.QUANTUM, MethodType.QUANTUM_GREEDY]
