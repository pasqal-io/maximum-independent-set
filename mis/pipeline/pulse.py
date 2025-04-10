from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from pulser import Pulse as PulserPulse
from pulser.waveforms import ConstantWaveform

from mis.pipeline.config import SolverConfig

import numpy as np
import networkx as nx

from .targets import Pulse, Register


class BasePulseShaper(ABC):
    """
    Abstract base class for generating pulse schedules based on a MIS problem.

    This class transforms the structure of a MISInstance into a quantum
    pulse sequence that can be applied to a physical register. The register
    is passed at the time of pulse generation, not during initialization.
    """

    @abstractmethod
    def generate(self, register: Register) -> Pulse:
        """
        Generate a pulse based on the problem and the provided register.

        Args:
            register (Register): The physical register layout.

        Returns:
            Pulse: A generated pulse object wrapping a Pulser pulse.
        """
        pass


class DefaultPulseShaper(BasePulseShaper):
    """
    A simple pulse shaper
    """

    @dataclass
    class Bounds:
        maximum_amplitude: float
        final_detuning: float

    def get_interactions(
            self,
            pos: np.ndarray,
            graph: nx.Graph,
            device: pulser.devices.Device
    ) -> tuple[list[float], list[float]]:
        """Calculate the interaction strengths for connected and disconnected nodes.

        Args:
            pos (np.ndarray): The position of the nodes.
            graph (nx.Graph): The associated graph.
            device (BaseDevice): Device used to calculate interaction coeff.

        Returns:
            tuple[list[float], list[float]]: Connected interactions, Disconnected interactions
        """

        def calculate_edge_interaction(edge: tuple[int, int]) -> float:
            pos_a, pos_b = pos[edge[0]], pos[edge[1]]
            return float(device.interaction_coeff / (euclidean(pos_a, pos_b) ** 6))

        connected = [calculate_edge_interaction(edge) for edge in graph.edges()]
        disconnected = [calculate_edge_interaction(edge)
                        for edge in nx.complement(graph).edges()]

        return connected, disconnected

    def calc_bounds(self,
                    reg: pulser.Register,
                    graph: nx.Graph,
                    device: pulser.devices.Device) -> Bounds:
        # FIXME: Sounds like this should go to pulse shaping
        _, disconnected = self.get_interactions(reg._coords, graph, device)
        u_min, u_max = self.interaction_bounds(reg._coords, graph, device)
        max_amp_device = device.channels["rydberg_global"].max_amp or np.inf
        maximum_amplitude = min(max_amp_device, u_max + 0.8 * (u_min - u_max))

        # Safely access graph attributes since graph is mandatory
        d_min = min(dict(graph.degree).values())
        d_max = max(dict(graph.degree).values())
        det_max_theory = (d_min / (d_min + 1)) * u_min
        det_min_theory = sum(sorted(disconnected)[-d_max:])
        det_final_theory = max([det_max_theory, det_min_theory])
        det_max_device = device.channels["rydberg_global"].max_abs_detuning or np.inf
        final_detuning = min(det_final_theory, det_max_device)

        return Bounds(
            maximum_amplitude=maximum_amplitude,
            final_detuning=final_detuning)

    def interaction_bounds(self,
                           pos: np.ndarray,
                           graph: nx.Graph,
                           device: pulser.devices.Device
                           ) -> tuple[float, float]:
        """Calculates U_min and U_max given the positions. It uses the edges of the
        graph. U_min corresponds to minimal energy of two nodes connected in the
        graph. U_max corresponds to maximal energy of two nodes NOT connected in
        the graph."""
        connected, disconnected = self.get_interactions(
            pos, graph, device)
        if len(connected) == 0:
            u_min = 0
        else:
            u_min = np.min(connected)
        if len(disconnected) == 0:
            u_max = np.inf
        else:
            u_max = np.max(disconnected)
        return u_min, u_max

    def generate(self, register: Register) -> Pulse:
        """
        Method to return a simple constant waveform pulse
        """
        wf = ConstantWaveform(duration=1000, value=1.0)
        pulser_pulse = PulserPulse.ConstantDetuning(amplitude=wf, detuning=0.0, phase=0.0)

        self.pulse = Pulse(pulse=pulser_pulse)
        return self.pulse

