from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from networkx.classes.reportviews import DegreeView
from pulser import AnalogDevice

from qoolqit import Drive, Register
from qoolqit.execution.backend import BaseBackend
from qoolqit.drive import WeightedDetuning
from mis.shared.graphs import WeightedPicker
from mis.shared.types import MISInstance, Weighting
from mis.pipeline.config import SolverConfig
from mis.pipeline.waveforms import InterpolatedWaveform

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

# Due to rounding errors, with some devices, running pulses with the max
# amplitude causes the sequence to be rejected. To avoid that, we multiply
# the max amplitude by AMP_SAFETY_FACTOR.
AMP_SAFETY_FACTOR = 0.99999


@dataclass
class BaseDriveShaper(ABC):
    """
    Abstract base class for generating drive schedules based on a MIS problem.

    This class transforms the structure of a MISInstance into a quantum
    drive that can be applied to a physical register. The register
    is passed at the time of drive generation, not during initialization.
    """

    duration_ns: int | None = None
    """The duration of the drive to be converted to a pulse at backend execution, 
        in nanoseconds.

    If unspecified, use the maximal duration for the device."""

    @abstractmethod
    def drive(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> Drive:
        """
        Generate a drive based on the problem and the provided register.

        Args:
            config: The configuration for this pulse.
            register: The physical register layout.

        Returns:
            Drive: A generated Drive object.
        """
        pass

    @abstractmethod
    def detuning(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> list[WeightedDetuning]:
        # By default, no detuning.
        return []


@dataclass
class DriveParameters:
    # Interaction strength for connected nodes.
    connected: list[float]

    # Interaction strength for disconnected nodes.
    disconnected: list[float]

    # Minimal energy between two connected nodes.
    u_min: float

    # Maximal energy between two disconnected nodes.
    maximum_amplitude: float

    # The duration of the pulse, in nanoseconds.
    duration_ns: float

    # The final detuning.
    final_detuning: float


class DefaultDriveShaper(BaseDriveShaper):
    """
    A simple pulse shaper.
    """

    def _calculate_parameters(
        self, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> DriveParameters:
        """
        Compute parameters shared between the pulse and the detunings.
        """
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.
        device = backend.device()
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.

        # Cache mapping node value -> node index.
        pos = list(register.qubits.values())
        assert len(pos) == len(graph)

        def calculate_edge_interaction(edge: tuple[int, int]) -> float:
            pos_a, pos_b = pos[edge[0]], pos[edge[1]]
            return float(device.interaction_coeff / (euclidean(pos_a, pos_b) ** 6))

        # Interaction strength for connected nodes.
        connected = [calculate_edge_interaction(edge) for edge in graph.edges()]

        # Interaction strength for disconnected nodes.
        disconnected = [calculate_edge_interaction(edge) for edge in nx.complement(graph).edges()]

        # Determine the minimal energy between two connected nodes.
        if len(connected) == 0:
            u_min = 0
        else:
            u_min = np.min(connected)

        # Determine the maximal energy between two disconnected nodes.
        max_amp_device = AMP_SAFETY_FACTOR * (device.channels["rydberg_global"].max_amp or np.inf)
        if len(disconnected) == 0:
            u_max = np.inf
            maximum_amplitude = max_amp_device
        else:
            u_max = np.max(disconnected)
            maximum_amplitude = min(max_amp_device, u_max + np.float16(0.8) * (u_min - u_max))
            # FIXME: Why 0.8?

        # Compute min/max degrees
        degree = graph.degree
        assert isinstance(degree, DegreeView)
        d_min = None
        d_max = None
        for _, deg in degree:
            assert isinstance(deg, int)
            if d_min is None or deg < d_min:
                d_min = deg
            if d_max is None or deg > d_max:
                d_max = deg
        assert d_min is not None
        assert d_max is not None
        assert isinstance(d_min, int)
        assert isinstance(d_max, int)
        det_max_theory = (d_min / (d_min + 1)) * u_min
        det_min_theory = sum(sorted(disconnected)[-d_max:])
        det_final_theory = max(det_max_theory, det_min_theory)
        det_max_device = device.channels["rydberg_global"].max_abs_detuning or np.inf
        final_detuning = min(det_final_theory, det_max_device)

        duration_ns = self.duration_ns
        if duration_ns is None:
            duration_ns = device.max_sequence_duration
        if duration_ns is None:
            # Last resort.
            duration_ns = AnalogDevice.max_sequence_duration
        assert duration_ns is not None

        return DriveParameters(
            duration_ns=duration_ns,
            connected=connected,
            disconnected=disconnected,
            u_min=u_min,
            maximum_amplitude=maximum_amplitude,
            final_detuning=final_detuning,
        )

    def drive(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> Drive:
        """
        Return a simple Drive with InterpolatedWaveform.
        """
        parameters = self._calculate_parameters(
            backend=backend, register=register, instance=instance
        )

        amplitude = InterpolatedWaveform(
            parameters.duration_ns, [1e-9, parameters.maximum_amplitude, 1e-9]
        )  # FIXME: This should be 0, investigate why it's 1e-9
        detuning = InterpolatedWaveform(
            parameters.duration_ns, [-parameters.final_detuning, 0, parameters.final_detuning]
        )
        rydberg_drive = Drive(amplitude=amplitude, detuning=detuning)

        return rydberg_drive

    def detuning(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> list[WeightedDetuning]:
        """
        Return detunings to be executed alongside the pulses.
        """
        if config.weighting == Weighting.UNWEIGHTED:
            return []

        parameters = self._calculate_parameters(
            register=register, backend=backend, instance=instance
        )

        # Normalize node weights to [0, 1]
        # FIXME: We assume that weights are >= 0, but we haven't checked that anywhere.
        max_weight: float = max(
            WeightedPicker.node_weight(instance.graph, x) for x in instance.graph
        )
        norm_node_weights = {
            register.qubit_ids[i]: 1 - WeightedPicker.node_weight(instance.graph, x) / max_weight
            for (i, x) in enumerate(instance.graph)
        }
        waveform = InterpolatedWaveform(
            parameters.duration_ns, values=[0, 0, -parameters.final_detuning]
        )

        # The constructor of InterpolatedWaveform does interesting metaprogramming
        # that mypy cannot follow.
        assert isinstance(waveform, InterpolatedWaveform)
        return [
            WeightedDetuning(
                weights=norm_node_weights,
                waveform=waveform,
            )
        ]
