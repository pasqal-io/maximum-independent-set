"""
Code emitted by compilation.

In practice, this code is a very thin layer around Pulser's representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

import pulser
import pulser.pulse
import pulser.register.weight_maps


@dataclass
class Pulse:
    """
    Specification of a laser pulse to be executed on a quantum device

    Attributes:
        pulse: The low-level Pulser pulse.
    """

    pulse: pulser.Pulse
    detuning_maps: (
        list[tuple[pulser.register.weight_maps.DetuningMap, pulser.waveforms.Waveform]] | None
    ) = None

    def draw(self) -> None:
        """
        Draw the shape of this laser pulse.
        """
        self.pulse.draw()


@dataclass
class Register:
    """
    Specification of a geometry of atoms to be executed on a quantum device

    Attributes:
        device: The quantum device targeted.
        register: The low-level Pulser register.
        graph: The graph laid out as register. Note that this is not
            necessarily the same graph as in MISInstance, as it may have
            been transformed by some intermediate steps.
    """

    device: pulser.devices.Device
    register: pulser.Register
    graph: nx.Graph

    def __post_init__(self) -> None:
        self.register = self.register.with_automatic_layout(self.device)

    def __len__(self) -> int:
        """
        The number of qubits in this register.
        """
        return len(self.register.qubits)

    def draw(self) -> None:
        """
        Draw the geometry of this register.
        """
        self.register.draw(blockade_radius=self.device.min_atom_distance + 0.01)
