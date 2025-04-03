from __future__ import annotations

from mis import QUBOInstance, QUBOSolution
from mis.config import SolverConfig
from mis.pipeline import (
    BaseSolver,
    Fixtures,
    Pulse,
    Register,
    get_embedder,
    get_pulse_shaper,
)


class MISSolver(BaseSolver):
    """
    Dispatcher that selects the appropriate solver (quantum or classical)
    based on the SolverConfig and delegates execution to it.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        super().__init__(instance, config)
        self._solver: BaseSolver

        if config.use_quantum:
            self._solver = MISSolverQuantum(instance, config)
        else:
            self._solver = MISSolverClassical(instance, config)

    def embedding(self) -> Register:
        return self._solver.embedding()

    def pulse(self, embedding: Register) -> Pulse:
        return self._solver.pulse(embedding)

    def solve(self) -> QUBOSolution:
        return self._solver.solve()


class MISSolverQuantum(BaseSolver):
    """
    Quantum solver that orchestrates the solving of a MISproblem using
    embedding, pulse shaping, and quantum execution pipelines.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        """
        Initialize the MISSolver with the given problem and configuration.

        Args:
            instance (QUBOInstance): The MISproblem to solve.
            config (SolverConfig): Solver settings including backend and device.
        """
        super().__init__(instance, config)

        self.fixtures = Fixtures(self.instance, self.config)
        self.embedder = get_embedder(self.instance, self.config)
        self.pulse_shaper = get_pulse_shaper(self.instance, self.config)

        self._register: Register | None = None
        self._pulse: Pulse | None = None
        self._solution: QUBOSolution | None = None

    def embedding(self) -> Register:
        """
        Generate a physical embedding (register) for the MISvariables.

        Returns:
            Register: Atom layout suitable for quantum hardware.
        """
        self._register = self.embedder.embed()
        return self._register

    def pulse(self, embedding: Register) -> Pulse:
        """
        Generate the pulse sequence based on the given embedding.

        Args:
            embedding (Register): The embedded register layout.

        Returns:
            Pulse: Pulse schedule for quantum execution.
        """
        self._pulse = self.pulse_shaper.generate(embedding)
        return self._pulse

    def solve(self) -> QUBOSolution:
        """
        Execute the full quantum pipeline: preprocess, embed, pulse, execute, postprocess.

        Returns:
            QUBOSolution: Final result after execution and postprocessing.
        """
        self.instance = self.fixtures.preprocess()

        embedding = self.embedding()
        pulse = self.pulse(embedding)
        execution_result = self.execute(pulse, embedding)

        best_bitstring = max(execution_result.items(), key=lambda x: x[1])[0]
        solution_vector = [int(bit) for bit in best_bitstring]
        energy = self.executor.register.device.evaluate_solution(solution_vector)

        solution = QUBOSolution(
            bitstrings=solution_vector,
            costs=energy,
            counts=execution_result,
        )

        self._solution = self.fixtures.postprocess(solution)
        return self._solution


class MISSolverClassical(BaseSolver):
    """
    Classical solver for MISproblems using brute-force search (for small instances).

    Intended for benchmarking or fallback when quantum execution is disabled.
    """

    def __init__(self, instance: QUBOInstance, config: SolverConfig):
        super().__init__(instance, config)

    def embedding(self) -> Register:
        return {}  # type: ignore[return-value]

    def pulse(self, embedding: Register) -> Pulse:
        return  # type: ignore[return-value]

    def solve(self) -> QUBOSolution:
        coeffs = self.instance.coefficients.cpu().numpy()
        best_solution = None
        best_energy = float("inf")
        # Solve the MISClassically
        return QUBOSolution(bitstrings=best_solution, costs=best_energy)
