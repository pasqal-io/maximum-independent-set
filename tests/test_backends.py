from __future__ import annotations

import datetime
import json
import random
from typing import Any, Counter, cast

import pulser
import pulser.pulse
import pytest
from pulser.waveforms import Waveform
from pasqal_cloud.utils.mock_server import BaseMockServer

from mis._backends import (
    BaseBackend,
    Detuning,
    EmuMPSBackend,
    EmuSVBackend,
    QutipBackend,
    RemoteEmuFREEBackend,
    RemoteEmuMPSBackend,
    RemoteJob,
    RemoteQPUBackend,
    get_backend,
)
from mis._backends.data import BackendConfig, BackendType, NamedDevice, QuantumProgram


def make_simple_program(backend: BaseBackend) -> QuantumProgram:
    return QuantumProgram(
        device=backend.device(),  # Use the default device
        register=pulser.Register({"q0": [0.0, 0.0], "q1": [10.0, 10.0]}),
        pulse=pulser.pulse.Pulse.ConstantPulse(
            duration=100,
            amplitude=0.5,
            phase=0.5,
            detuning=0,
        ),
        detunings=[
            Detuning(
                weights={"q0": 0.5, "q1": 0.5},
                waveform=cast(
                    Waveform, pulser.waveforms.ConstantWaveform(duration=100, value=-0.5)
                ),
            )
        ],
    )


local_backends: list[tuple[type[BaseBackend], BackendType]] = [
    (QutipBackend, BackendType.QUTIP),
    (EmuMPSBackend, BackendType.EMU_MPS),
    (EmuSVBackend, BackendType.EMU_SV),
]

remote_backends: list[tuple[type[BaseBackend], BackendType]] = [
    (RemoteEmuMPSBackend, BackendType.REMOTE_EMUMPS),
    (RemoteEmuFREEBackend, BackendType.REMOTE_EMUFREE),
    (RemoteQPUBackend, BackendType.REMOTE_QPU),
]

all_backends = local_backends + remote_backends


def test_get_backend() -> None:
    """Test that `get_backend()` successfully instantiates backends of the right type."""
    for cls, type in all_backends:
        backend_config = BackendConfig(backend=type)
        backend = get_backend(backend_config)
        assert isinstance(backend, cls)


@pytest.mark.parametrize("backend_kind", local_backends)
@pytest.mark.parametrize("num_shots", [1, 10, 100])
def test_local_execute(backend_kind: tuple[type[BaseBackend], BackendType], num_shots: int) -> None:
    """Test that we can run locally a simple quantum program works."""
    backend_config = BackendConfig(
        backend=backend_kind[1],
    )
    backend = get_backend(backend_config)
    program = make_simple_program(backend)
    result = backend.run(program, num_shots)
    assert len(result.counts) >= 1
    assert len(result) == num_shots
    for k in result.counts.keys():
        assert len(k) == 2  # Two qubits


@pytest.mark.parametrize("backend_kind", remote_backends)
@pytest.mark.parametrize("num_shots", [1, 10, 100])
def test_remote_execute(
    backend_kind: tuple[type[BaseBackend], BackendType], num_shots: int
) -> None:
    """Test that we can run remotely a simple quantum program works."""

    # for some reason, pytest requires class inside
    class MyMockServer(BaseMockServer):
        def __init__(self) -> None:
            super().__init__()
            self.mocker.get("http://example.com/results/my-results", json=self.my_results)
            self._iterations = 0
            self._start = str(datetime.datetime.now())
            self._runs = None

        def endpoint_get_devices_specs(self, request: Any, context: Any, matches: list[str]) -> Any:
            """Return a basic device called `MY_DEVICE`."""
            return {
                "data": {"MY_DEVICE": pulser.DigitalAnalogDevice.to_abstract_repr()},
            }

        def endpoint_get_devices_public_specs(
            self, request: Any, context: Any, matches: list[str]
        ) -> Any:
            """Return a basic device called `MY_DEVICE`."""
            return {
                "data": [
                    {
                        "device_type": "MY_DEVICE",
                        "specs": pulser.DigitalAnalogDevice.to_abstract_repr(),
                    },
                ],
            }

        def endpoint_post_batch(self, request: Any, context: Any, matches: list[str]) -> Any:
            batch: dict[str, Any] = json.loads(request.text)
            data = self.add_batch(batch)
            return {
                "data": data,
            }

        def endpoint_get_batch(self, request: Any, context: Any, matches: list[str]) -> Any:
            """Mock for GET /api/v1/batches/{batch_id}."""
            assert matches[0] == "my-mock-batch-id"

            if self._iterations < 5:
                self._iterations += 1
                # Initially, respond that the batch is pending.
                return {
                    "data": {
                        "open": False,
                        "complete": False,
                        "created_at": self._start,
                        "updated_at": str(datetime.datetime.now()),
                        "project_id": "my-project-id",
                        "id": "my-mock-batch-id",
                        "user_id": "my-user-id",
                        "status": "PENDING",
                        "ordered_jobs": [],
                        "device_type": "DigitalAnalogDevice",
                    }
                }

            return {
                "data": {
                    "open": False,
                    "complete": False,
                    "created_at": self._start,
                    "updated_at": str(datetime.datetime.now()),
                    "project_id": "my-project-id",
                    "id": "my-mock-batch-id",
                    "user_id": "my-user-id",
                    "status": "DONE",
                    "ordered_jobs": [],
                    "device_type": "DigitalAnalogDevice",
                }
            }

        def endpoint_get_jobs(self, request: Any, context: Any, matches: list[str]) -> Any:
            return {
                "code": 200,
                "data": [
                    {
                        "id": "my-job-id",
                        "parent-id": "my-parent-id",
                        "status": "DONE",
                        "runs": self._runs,
                        "batch_id": "my-match-id",
                        "project_id": "my-project-id",
                        "created_at": self._start,
                        "updated_at": str(datetime.datetime.now()),
                        "creation_order": 0,
                    }
                ],
                "message": "OK.",
                "status": "success",
            }

        def add_batch(self, batch: Any) -> Any:
            assert batch["project_id"] == "my-project-id"
            self._runs = batch["jobs"][0]["runs"]
            self._register = json.loads(batch["sequence_builder"])["register"]
            return {
                "open": False,
                "complete": False,
                "created_at": self._start,
                "updated_at": str(datetime.datetime.now()),
                "project_id": batch["project_id"],
                "id": "my-mock-batch-id",
                "user_id": "my-user-id",
                "status": "PENDING",
                "ordered_jobs": [],
                "device_type": "DigitalAnalogDevice",
            }

        def my_results(self, request: Any, context: Any) -> Any:
            counter: Counter[str] = Counter()
            assert isinstance(self._runs, int)
            for _ in range(self._runs):
                # Generate key
                bitstr: str = ""
                for _ in range(len(self._register)):
                    bitstr += str(random.randrange(0, 2))
                counter[bitstr] += 1
            return {"counter": counter}

        def endpoint_get_job_results(self, request: Any, context: Any, matches: list[str]) -> Any:
            return {
                "data": {
                    "results_link": "http://example.com/results/my-results",
                },
            }

    backend_config = BackendConfig(
        backend=backend_kind[1],
        username="my_username",
        password="my_password",
        project_id="my-project-id",
        device=NamedDevice("MY_DEVICE"),
    )
    with MyMockServer():
        backend = get_backend(backend_config)
        device = backend.device()
        assert device.name == "DigitalAnalogDevice"
        program = make_simple_program(backend)
        job = backend.submit(program, num_shots)
        assert job.id == "my-mock-batch-id"
        assert isinstance(job, RemoteJob)
        job.sleep_duration_sec = 0.1  # Speed up test
        result = job.wait()
        assert len(result.counts) >= 1
        assert len(result) == num_shots
        for k in result.counts.keys():
            assert len(k) == 2  # Two qubits
