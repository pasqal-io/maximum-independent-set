"""
Tools to prepare the geometry (register) of atoms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


import pulser
from pulser import Register
from mis._backends.backends import BaseBackend

from mis.shared.types import (
    MISInstance,
)
from mis.pipeline.config import SolverConfig

from .layout import Layout


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Pasqal/Pulser devices.
    """

    @abstractmethod
    def embed(self, instance: MISInstance, config: SolverConfig, backend: BaseBackend) -> Register:
        """
        Creates a layout of atoms as the register.

        Returns:
            Register: The register.
        """
        pass


class DefaultEmbedder(BaseEmbedder):
    """
    A simple embedder
    """

    def embed(self, instance: MISInstance, config: SolverConfig, backend: BaseBackend) -> Register:
        device = backend.device()
        assert device is not None

        # Use Layout helper to get rescaled coordinates and interaction graph
        layout = Layout.from_device(data=instance, device=device)

        # Finally, prepare register.
        return pulser.register.Register(
            qubits={f"q{node}": pos for (node, pos) in layout.coords.items()}
        )


class OptimizedEmbedder(BaseEmbedder):
    """
    An embedder using constrained optimization
    (via Sequential Least Squares Programming (SLSQP))
    to find coordinates that respect device constrained
    after the DefaultEmbedder.

    We try at most 10 times to run the optimization to find
    a suitable embedding.
    """

    def embed(self, instance: MISInstance, config: SolverConfig, backend: BaseBackend) -> Register:
        import numpy as np
        from scipy.optimize import minimize, NonlinearConstraint

        device = backend.device()
        assert device is not None
        if not (hasattr(device, "min_atom_distance") and hasattr(device, "max_radial_distance")):
            raise ValueError(
                "OptimizedEmbedder does not apply if device "
                "has no min_atom_distance and max_radial_distance constraints"
            )

        register = DefaultEmbedder().embed(instance, config, backend)

        nb_tries = 0
        while nb_tries < 10:
            nb_tries += 1
            coords = np.array(list(register.qubits.values()))
            n = coords.shape[0]
            x0 = coords.flatten()

            center = np.mean(coords, axis=0)
            # We multiply by factors to be (reasonably) certain that we're slightly
            # within bounds.
            min_atom_distance = 1.0000001 * device.min_atom_distance
            max_radial_distance = 0.0000099 * device.max_radial_distance

            # Objective: keep positions near original
            def objective(x: np.ndarray) -> float:
                return float(np.sum((x - x0) ** 2))

            # Constraint: all pairwise distances ≥ min_atom_distance
            def pairwise_constraints(x: np.ndarray) -> np.ndarray:
                pts = x.reshape((n, 2))
                vals = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(pts[i] - pts[j])
                        vals.append(d - min_atom_distance)
                return np.array(vals)

            # Constraint: all points within max_radial_distance
            def radial_constraints(x: np.ndarray) -> np.ndarray:
                pts = x.reshape((n, 2))
                dists = np.linalg.norm(pts - center, axis=1)
                return max_radial_distance - dists

            cons = [
                NonlinearConstraint(pairwise_constraints, 0, np.inf),
                NonlinearConstraint(radial_constraints, 0, np.inf),
            ]

            res = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints=cons,
                options={"maxiter": 1000, "ftol": 1e-6},
            )
            coords = res.x.reshape((n, 2))
            qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}
            register = Register(qubits)
            try:
                device.validate_register(register)
                break
            except Exception:
                continue
        self._nb_tries = nb_tries
        return register
