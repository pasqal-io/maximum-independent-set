from __future__ import annotations

import abc
import asyncio
from enum import Enum
from typing import Callable, Generic, TypeVar


class Status(str, Enum):
    IN_PROGRESS = "in-progress"
    SUCCESS = "success"
    FAILURE = "failure"


Result = TypeVar("Result")
Transformed = TypeVar("Transformed")


class Execution(abc.ABC, Generic[Result]):
    """
    A task being sent to a quantum device and which
    may need some time before it is completed.
    """

    @abc.abstractmethod
    def result(self) -> Result: ...

    @abc.abstractmethod
    async def wait(self) -> Result: ...

    @abc.abstractmethod
    def status(self) -> Status: ...

    def map(self, transform: Callable[[Result], Transformed]) -> Execution[Transformed]:
        """
        Apply a transformation to the result once it is
        complete.
        """
        return MappedExecution(self, transform)

    @classmethod
    def success(cls, result: Result) -> Execution[Result]:
        """
        Shortcut to return a result that has already succeeded.
        """
        return SuccessfulExecution(result)


class WaitingExecution(Execution[Result]):
    """
    A task being sent to a quantum device and which
    definitely needs some time before it is completed.

    Unless you're implementing new executors, you're probably
    not interested in this class.
    """

    def __init__(self, sleep_sec: int):
        self._sleep_sec = sleep_sec

    async def wait(self) -> Result:
        while self.status() is Status.IN_PROGRESS:
            await asyncio.sleep(self._sleep_sec)
        return self.result()


class MappedExecution(Execution[Transformed]):
    """
    The result of calling `map` on an `Execution`.

    Unless you're implementing new executors, you're probably
    not interested in this class.
    """

    def __init__(self, origin: Execution[Result], transform: Callable[[Result], Transformed]):
        self._cache_filled = False
        self._origin = origin
        self._transform = transform

    def result(self) -> Transformed:
        original = self._origin.result()
        return self._transform(original)

    def status(self) -> Status:
        return self._origin.status()

    async def wait(self) -> Transformed:
        original = await self._origin.wait()
        return self._transform(original)


class SuccessfulExecution(Execution[Result]):
    """
    An execution that is already completed.

    Unless you're implementing new executors, you're probably
    not interested in this class.
    """

    def __init__(self, result: Result):
        self._result = result

    def status(self) -> Status:
        return Status.SUCCESS

    async def wait(self) -> Result:
        # No need to wait.
        return self.result()

    def result(self) -> Result:
        return self._result

    def map(self, transform: Callable[[Result], Transformed]) -> Execution[Transformed]:
        # Since we know that we're not waiting for anything,
        # we can perform the `map` call immediately.
        mapped = transform(self.result())
        return SuccessfulExecution(result=mapped)
