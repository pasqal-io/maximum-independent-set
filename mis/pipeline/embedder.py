"""
Tools to prepare the geometry (register) of atoms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pulser import Register as PulserRegister
from qoolqit import Register

from mis.shared.types import (
    MISInstance,
)
from mis.pipeline.config import SolverConfig

from .layout import Layout


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.

    Prepares the geometry (register) of atoms based on the MISinstance.
    Returns a Register compatible with Qoolqit devices.
    """

    @abstractmethod
    def embed(self, instance: MISInstance, config: SolverConfig) -> Register:
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

    def embed(self, instance: MISInstance, config: SolverConfig) -> Register:
        device = config.device
        assert device is not None

        # Use Layout helper to get rescaled coordinates and interaction graph
        layout = Layout.from_device(data=instance, device=device)

        # Finally, prepare register.
        conversion_factor = device.converter.factors[2]
        return Register(
            qubits={f"q{node}": pos / conversion_factor for (node, pos) in layout.coords.items()}
        )


class OptimizedEmbedder(BaseEmbedder):
    """
    An embedder using constrained optimization
    (via Sequential Least Squares Programming (SLSQP))
    to find coordinates that respect device constrained
    after the the coordinates found with Layout.

    We try at most 10 times to run the optimization to find
    a suitable embedding.
    """

    def embed(self, instance: MISInstance, config: SolverConfig) -> Register:
        import numpy as np
        from scipy.optimize import minimize, NonlinearConstraint

        layout = Layout.from_device(data=instance, device=config.device)
        pulser_device = config.device._device

        coords = np.array(list(layout.coords.values()))
        n = coords.shape[0]

        nb_tries = 0
        while nb_tries < 10:
            nb_tries += 1

            center = np.mean(coords, axis=0)

            # initial coordinates for optimizer
            x0 = coords.flatten()

            # We multiply by factors to be (reasonably) certain that we're slightly
            # within bounds.
            min_atom_distance = 1.0000001 * pulser_device.min_atom_distance
            max_radial_distance = 0.0000099 * pulser_device.max_radial_distance

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
            pulser_register = PulserRegister(qubits)
            try:
                pulser_device.validate_register(pulser_register)
                break
            except Exception:
                continue
        self._nb_tries = nb_tries

        # Finally, prepare register.
        conversion_factor = config.device.converter.factors[2]
        return Register(qubits={q: pos / conversion_factor for (q, pos) in qubits.items()})
