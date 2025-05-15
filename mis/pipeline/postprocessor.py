import abc

from mis.shared.types import MISSolution


class BasePostprocessor(abc.ABC):
    @abc.abstractmethod
    def postprocess(self, solution: MISSolution) -> MISSolution | None:
        """
        Post-process a solution, typically to improve its quality.

        May return `None` if the solution is deemed unacceptable.
        """
        ...
