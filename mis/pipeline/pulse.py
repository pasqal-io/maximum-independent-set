from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from networkx.classes.reportviews import DegreeView
from pulser import InterpolatedWaveform, Pulse, Register

from qoolqit._solvers.backends import BaseBackend
from mis.shared.types import MISInstance
from mis.pipeline.config import SolverConfig

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean


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
    def generate(
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


class DefaultPulseShaper(BasePulseShaper):
    """
    A simple pulse shaper.
    """

    def generate(
        self, config: SolverConfig, register: Register, backend: BaseBackend, instance: MISInstance
    ) -> Pulse:
        """
        Return a simple constant waveform pulse
        """

        device = backend.device()
        graph = instance.graph  # Guaranteed to be consecutive integers starting from 0.

        # Cache mapping node value -> node index.
        pos = register.sorted_coords

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
        if len(disconnected) == 0:
            u_max = np.inf
        else:
            u_max = np.max(disconnected)

        max_amp_device = device.channels["rydberg_global"].max_amp or np.inf
        maximum_amplitude = min(max_amp_device, u_max + 0.8 * (u_min - u_max))
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

        amplitude = InterpolatedWaveform(
            duration_us, [1e-9, maximum_amplitude, 1e-9]
        )  # FIXME: This should be 0, investigate why it's 1e-9
        detuning = InterpolatedWaveform(duration_us, [-final_detuning, 0, final_detuning])
        rydberg_pulse = Pulse(amplitude, detuning, 0)
        # Pulser overrides PulserPulse.__new__ with an exotic type, so we need
        # to help mypy.
        assert isinstance(rydberg_pulse, Pulse)

        return rydberg_pulse
