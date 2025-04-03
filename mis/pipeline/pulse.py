from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod

from pulser import Pulse as PulserPulse
from pulser.waveforms import ConstantWaveform

from mis.config import SolverConfig

from .targets import Pulse, Register


class BasePulseShaper(ABC):
    """
    Abstract base class for generating pulse schedules based on a MIS problem.

    This class transforms the structure of a MISInstance into a quantum
    pulse sequence that can be applied to a physical register. The register
    is passed at the time of pulse generation, not during initialization.
    """

    def __init__(self, instance: Any, config: SolverConfig):
        """
        Initialize the pulse shaping module with a MIS instance.

        Args:
            instance (MISInstance): The MIS problem instance.
        """
        self.instance = instance
        self.config: SolverConfig = config
        self.pulse: Pulse | None = None

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


class FirstPulseShaper(BasePulseShaper):
    """
    A simple pulse shaper
    """

    def generate(self, register: Register) -> Pulse:
        """
        Method to return a simple constant waveform pulse
        """
        wf = ConstantWaveform(duration=1000, value=1.0)
        pulser_pulse = PulserPulse.ConstantDetuning(amplitude=wf, detuning=0.0, phase=0.0)

        self.pulse = Pulse(pulse=pulser_pulse)
        return self.pulse


def get_pulse_shaper(instance: Any, config: SolverConfig) -> BasePulseShaper:
    """
    Method that returns the correct PulseShaper based on configuration.
    The correct pulse shaping method can be identified using the config, and an
    object of this pulseshaper can be returned using this function.

    Args:
        instance (MISInstance): The MIS problem to embed.
        config (Device): The quantum device to target.

    Returns:
        (BasePulseShaper): The representative Pulse Shaper object.
    """

    return FirstPulseShaper(instance, config)
