from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from networkx.classes.reportviews import DegreeView
from pulser import AnalogDevice, InterpolatedWaveform, Pulse, Register

from qoolqit._solvers.backends import BaseBackend
from qoolqit._solvers import Detuning
from mis.shared.graphs import WeightedPicker
from mis.shared.types import MISInstance, Weighting
from mis.pipeline.config import SolverConfig

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

# Due to rounding errors, with some devices, running pulses with the max
# amplitude causes the sequence to be rejected. To avoid that, we multiply
# the max amplitude by AMP_SAFETY_FACTOR.
AMP_SAFETY_FACTOR = 0.99999


@dataclass
class BasePulseShaper(ABC):
    """
    Abstract base class for generating pulse schedules based on a MIS problem.

    This class transforms the structure of a MISInstance into a quantum
    pulse sequence that can be applied to a physical register. The register
    is passed at the time of pulse generation, not during initialization.
    """

    duration_us: int | None = None
    """The duration of the pulse, in microseconds.

    If unspecified, use the maximal duration for the device."""

    @abstractmethod
    def pulse(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> Pulse:
        """
        Generate a pulse based on the problem and the provided register.

        Args:
            config: The configuration for this pulse.
            register: The physical register layout.

        Returns:
            Pulse: A generated pulse object wrapping a Pulser pulse.
        """
        pass

    @abstractmethod
    def detuning(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> list[Detuning]:
        # By default, no detuning.
        return []


@dataclass
class _Parameters:
    final_detuning: float
    duration_us: int
    maximum_amplitude: float


class DefaultPulseShaper(BasePulseShaper):
    """
    A simple pulse shaper.
    """

    def _calculate_parameters(
        self, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> _Parameters:
        """
        Compute parameters shared between the pulse and the detunings.
        """
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.
        device = backend.device()

        # Cache mapping node value -> node index.
        pos = register.sorted_coords
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

        duration_us = self.duration_us
        if duration_us is None:
            duration_us = device.max_sequence_duration
        if duration_us is None:
            # Last resort.
            duration_us = AnalogDevice.max_sequence_duration
        assert duration_us is not None

        return _Parameters(
            final_detuning=final_detuning,
            duration_us=duration_us,
            maximum_amplitude=maximum_amplitude,
        )

    def detuning(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> list[Detuning]:
        """
        Return a simple constant waveform pulse
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
            parameters.duration_us, values=[0, 0, -parameters.final_detuning]
        )

        # The constructor of InterpolatedWaveform does interesting metaprogramming
        # that mypy cannot follow.
        assert isinstance(waveform, InterpolatedWaveform)
        return [
            Detuning(
                weights=norm_node_weights,
                waveform=waveform,
            )
        ]

    def pulse(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> Pulse:
        """
        Return a simple constant waveform pulse
        """
        parameters = self._calculate_parameters(
            backend=backend, register=register, instance=instance
        )

        amplitude = InterpolatedWaveform(
            parameters.duration_us, [1e-9, parameters.maximum_amplitude, 1e-9]
        )  # FIXME: This should be 0, investigate why it's 1e-9
        detuning = InterpolatedWaveform(
            parameters.duration_us, [-parameters.final_detuning, 0, parameters.final_detuning]
        )
        rydberg_pulse = Pulse(amplitude, detuning, 0)
        # Pulser overrides PulserPulse.__new__ with an exotic type, so we need
        # to help mypy.
        assert isinstance(rydberg_pulse, Pulse)

        return rydberg_pulse
