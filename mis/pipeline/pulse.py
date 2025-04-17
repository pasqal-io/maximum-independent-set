from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from networkx.classes.reportviews import DegreeView
from pulser import InterpolatedWaveform, Pulse as PulserPulse
from pulser.devices import Device

from mis.pipeline.config import SolverConfig

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean

from .targets import Pulse, Register


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
    def generate(self, config: SolverConfig, register: Register) -> Pulse:
        """
        Generate a pulse based on the problem and the provided register.

        Args:
            config: The configuration for this pulse.
            register: The physical register layout.

        Returns:
            Pulse: A generated pulse object wrapping a Pulser pulse.
        """
        pass


@dataclass
class _Bounds:
    maximum_amplitude: float
    final_detuning: float


class DefaultPulseShaper(BasePulseShaper):
    """
    A simple pulse shaper.
    """

    def _get_interactions(
        self, pos: np.ndarray, graph: nx.Graph, device: Device
    ) -> tuple[list[float], list[float]]:
        """Calculate the interaction strengths for connected and disconnected
            nodes.

        Args:
            pos (np.ndarray): The position of the nodes.
            graph (nx.Graph): The associated graph.
            device (BaseDevice): Device used to calculate interaction coeff.

        Returns:
            tuple[list[float], list[float]]: Connected interactions,
                Disconnected interactions
        """

        def calculate_edge_interaction(edge: tuple[int, int]) -> float:
            pos_a, pos_b = pos[edge[0]], pos[edge[1]]
            return float(device.interaction_coeff / (euclidean(pos_a, pos_b) ** 6))

        connected = [calculate_edge_interaction(edge) for edge in graph.edges()]
        disconnected = [calculate_edge_interaction(edge) for edge in nx.complement(graph).edges()]

        return connected, disconnected

    def _calc_bounds(self, reg: Register, device: Device) -> _Bounds:
        _, disconnected = self._get_interactions(
            pos=reg.register.sorted_coords, graph=reg.graph, device=device
        )
        u_min, u_max = self._interaction_bounds(
            pos=reg.register.sorted_coords, graph=reg.graph, device=device
        )
        max_amp_device = device.channels["rydberg_global"].max_amp or np.inf
        maximum_amplitude = min(max_amp_device, u_max + 0.8 * (u_min - u_max))

        degree = reg.graph.degree
        assert isinstance(degree, DegreeView)
        d_min = min(dict(degree).values())
        d_max = max(dict(degree).values())
        det_max_theory = (d_min / (d_min + 1)) * u_min
        det_min_theory = sum(sorted(disconnected)[-d_max:])
        det_final_theory = max([det_max_theory, det_min_theory])
        det_max_device = device.channels["rydberg_global"].max_abs_detuning or np.inf
        final_detuning = min(det_final_theory, det_max_device)

        return _Bounds(maximum_amplitude=maximum_amplitude, final_detuning=final_detuning)

    def _interaction_bounds(
        self, pos: np.ndarray, graph: nx.Graph, device: Device
    ) -> tuple[float, float]:
        """Calculates U_min and U_max given the positions. It uses the edges
        of the graph. U_min corresponds to minimal energy of two nodes
        connected in the graph. U_max corresponds to maximal energy of two
        nodes NOT connected in the graph."""
        connected, disconnected = self._get_interactions(pos, graph, device)
        if len(connected) == 0:
            u_min = 0
        else:
            u_min = np.min(connected)
        if len(disconnected) == 0:
            u_max = np.inf
        else:
            u_max = np.max(disconnected)
        return u_min, u_max

    def generate(self, config: SolverConfig, register: Register) -> Pulse:
        """
        Method to return a simple constant waveform pulse
        """
        device = config.device
        assert device is not None

        duration_us = self.duration_us
        if duration_us is None:
            duration_us = device.max_sequence_duration

        bounds = self._calc_bounds(reg=register, device=device)

        amplitude = InterpolatedWaveform(
            duration_us, [1e-9, bounds.maximum_amplitude, 1e-9]
        )  # FIXME: This should be 0, investigate why it's 1e-9
        detuning = InterpolatedWaveform(
            duration_us, [-bounds.final_detuning, 0, bounds.final_detuning]
        )
        rydberg_pulse = PulserPulse(amplitude, detuning, 0)
        # Pulser overrides PulserPulse.__new__ with an exotic type, so we need
        # to help mypy.
        assert isinstance(rydberg_pulse, PulserPulse)

        return Pulse(pulse=rydberg_pulse)
