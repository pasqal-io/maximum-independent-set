from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from networkx.classes.reportviews import DegreeView
from pulser import AnalogDevice

from qoolqit import Drive, Register
from qoolqit.devices import Device
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
class DriveParameters:
    # Interaction strength for connected nodes.
    connected: list[float]

    # Interaction strength for disconnected nodes.
    disconnected: list[float]

    # Minimal energy between two connected nodes.
    u_min: float

    # Maximal energy between two disconnected nodes.
    maximum_amplitude: float

    # The duration of the drive.
    duration: float

    # The final detuning.
    final_detuning: float


@dataclass
class BaseDriveShaper(ABC):
    """
    Abstract base class for generating drive schedules based on a MIS problem.

    This class transforms the structure of a MISInstance into a quantum
    drive that can be applied to a physical register. The register
    is passed at the time of drive generation, not during initialization.
    """

    duration: float | None = None
    """The duration of the drive to be converted to a pulse at backend execution.

    If unspecified, use the maximal duration for the device."""

    @abstractmethod
    def drive(self, config: SolverConfig, register: Register, instance: MISInstance) -> Drive:
        """
        Generate a drive based on the problem and the provided register.

        Args:
            config: The configuration for this drive.
            register: The physical register layout.
            instance: MIS instance.

        Returns:
            Drive: A generated Drive object.
        """
        pass

    @abstractmethod
    def weighted_detuning(
        self,
        config: SolverConfig,
        register: Register,
        instance: MISInstance,
        parameters: DriveParameters,
    ) -> list[WeightedDetuning] | None:
        # By default, no detuning.
        return None


class DefaultDriveShaper(BaseDriveShaper):
    """
    A simple drive shaper.
    """

    def _calculate_parameters(
        self, register: Register, device: Device, instance: MISInstance
    ) -> DriveParameters:
        """
        Compute parameters of the drive with the weighted detunings.
        """
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.

        # Cache mapping node value -> node index.
        pos = list(register.qubits.values())
        assert len(pos) == len(graph)

        def calculate_edge_interaction(edge: tuple[int, int]) -> float:
            pos_a, pos_b = pos[edge[0]], pos[edge[1]]
            return float(device._device.interaction_coeff / (euclidean(pos_a, pos_b) ** 6))

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
        max_amp_device = AMP_SAFETY_FACTOR * (
            device._device.channels["rydberg_global"].max_amp or np.inf
        )
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
        det_max_device = device._device.channels["rydberg_global"].max_abs_detuning or np.inf
        final_detuning = min(det_final_theory, det_max_device)

        duration_ns = self.duration or 0.0
        if duration_ns == 0.0:
            duration_ns = device._device.max_sequence_duration or 0.0
        if duration_ns == 0.0:
            # Last resort.
            duration_ns = AnalogDevice.max_sequence_duration or 0.0
        assert duration_ns > 0

        # for conversions to qoolqit
        TIME, _, _ = device.converter.factors
        duration_ns /= TIME
        maximum_amplitude /= TIME
        final_detuning /= TIME

        return DriveParameters(
            duration=duration_ns,
            connected=connected,
            disconnected=disconnected,
            u_min=u_min,
            maximum_amplitude=maximum_amplitude,
            final_detuning=final_detuning,
        )

    def drive(self, config: SolverConfig, register: Register, instance: MISInstance) -> Drive:
        """
        Return a simple Drive with InterpolatedWaveform.
        """
        parameters = self._calculate_parameters(
            device=config.device, register=register, instance=instance
        )

        amplitude = InterpolatedWaveform(
            parameters.duration, [1e-9, parameters.maximum_amplitude, 1e-9]
        )  # FIXME: This should be 0, investigate why it's 1e-9
        detuning = InterpolatedWaveform(
            parameters.duration, [-parameters.final_detuning, 0, parameters.final_detuning]
        )
        rydberg_drive = Drive(
            amplitude=amplitude,
            detuning=detuning,
            weighted_detunings=self.weighted_detuning(
                config=config, register=register, instance=instance, parameters=parameters
            ),
        )

        return rydberg_drive

    def weighted_detuning(
        self,
        config: SolverConfig,
        register: Register,
        instance: MISInstance,
        parameters: DriveParameters,
    ) -> list[WeightedDetuning] | None:
        """Return weighted detunings to be executed within the drive."""

        if config.weighting == Weighting.UNWEIGHTED:
            return None

        if len(list(config.device._device.dmm_channels.keys())) == 0:
            return None

        # Normalize node weights to [0, 1]
        # FIXME: We assume that weights are >= 0, but we haven't checked that anywhere.
        max_weight: float = max(
            WeightedPicker.node_weight(instance.graph, x) for x in instance.graph
        )
        norm_node_weights = {
            register.qubits_ids[i]: 1 - WeightedPicker.node_weight(instance.graph, x) / max_weight
            for (i, x) in enumerate(instance.graph)
        }
        waveform = InterpolatedWaveform(
            parameters.duration, values=[0, 0, -parameters.final_detuning]
        )
        return [
            WeightedDetuning(
                weights=norm_node_weights,
                waveform=waveform,
            )
        ]
