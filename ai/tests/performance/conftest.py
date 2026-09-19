"""Shared release-performance fixtures (MAT test plan, Task 8).

Two serving surfaces, both identical to production:

- In-process nodes run the real composition root (``create_app``) with the
  production ``load_model_from_artifact`` loader. Activation performs exactly
  the two calls the production lifespan performs (``startup_load`` +
  ``startup_normalization``) and is pinned to ``Settings()`` defaults, whose
  ``model_backend="baseline"`` is the production serving backend.
- Subprocess nodes spawn the real ``main:app`` through uvicorn on 127.0.0.1
  and read ``VmRSS`` from ``/proc/<pid>/status``.

The 1,014-input corpus is loaded through the exact loader used by
``scripts/qa_verify_corpus_equivalence.py`` so the throughput/memory nodes score
the same frozen corpus that equivalence evidence covers.
"""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI

from src.api.dependencies import (
    RuntimeState,
    load_model_from_artifact,
    startup_load,
    startup_normalization,
)
from src.api.server import create_app
from src.core.settings import Settings

AI_ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_S = 30.0
CORPUS_TOTAL = 1_028
CORPUS_SIZES = {"showcase": 54, "challenge": 74, "dataset": 900}


@dataclass
class SubprocessServer:
    """A spawned uvicorn ``main:app`` process plus the port it listens on."""

    process: subprocess.Popen[bytes]
    port: int

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def free_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def read_vmrss_kb(pid: int) -> int:
    """Peak-current resident set of ``pid`` in kB, from /proc (Linux)."""
    with Path(f"/proc/{pid}/status").open(encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    message = f"VmRSS entry missing for pid {pid}"
    raise RuntimeError(message)


def spawn_baseline_server() -> SubprocessServer:
    """Start the real ``main:app`` via uvicorn on a free loopback port."""
    port = free_tcp_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=AI_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return SubprocessServer(process=process, port=port)


def stop_server(server: SubprocessServer) -> None:
    """Terminate the subprocess; escalate to kill only if terminate hangs."""
    server.process.terminate()
    try:
        server.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.process.kill()
        server.process.wait(timeout=10)


def wait_until_ready(
    port: int, started_ns: int | None = None, timeout_s: float = STARTUP_TIMEOUT_S
) -> float:
    """Poll /health-check until 200; return seconds elapsed since ``started_ns``.

    Polling is transport-only: the asserted quantity is the measured
    startup time recorded by the caller, never a sleep duration.
    """
    if started_ns is None:
        started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + int(timeout_s * 1_000_000_000)
    while time.perf_counter_ns() < deadline_ns:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health-check", timeout=1.0
            ) as response:
                if response.status == 200:
                    return (time.perf_counter_ns() - started_ns) / 1_000_000_000
        except (OSError, urllib.error.URLError):
            time.sleep(0.01)
    message = f"server on port {port} did not reach /health-check 200 within {timeout_s}s"
    raise AssertionError(message)


@pytest.fixture(scope="session")
def corpus_inputs() -> list[tuple[str, str, str]]:
    """The frozen 1,014 corpus rows as (source, case_id, text), fixed order."""
    from scripts.qa_verify_corpus_equivalence import load_corpus

    grouped: dict[str, list[tuple[str, str]]] = load_corpus()
    inputs: list[tuple[str, str, str]] = []
    for source in ("showcase", "challenge", "dataset"):
        for case_id, text in grouped[source]:
            inputs.append((source, case_id, text))
    counts = Counter(source for source, _, _ in inputs)
    assert len(inputs) == CORPUS_TOTAL, counts
    assert dict(counts) == CORPUS_SIZES, counts
    return inputs


@pytest.fixture(scope="session")
def baseline_app() -> FastAPI:
    """Production app serving the verified baseline backend, ready for traffic.

    This is the production lifespan body (``src/api/server.py``) executed
    directly so in-process ASGI transport can serve without an event-loop-
    bound session fixture: same loader, same locked catalog, backend pinned
    to ``baseline`` by ``Settings`` defaults exactly as production serving.
    """
    app = create_app(Settings(), load_model_from_artifact)
    state: RuntimeState = app.state.runtime
    startup_load(state, Settings(), load_model_from_artifact)
    startup_normalization(state, Settings())
    assert state.ready, state.startup_error
    assert not state.degraded
    assert state.backend == "linear"
    return app


@pytest.fixture(scope="session")
def baseline_subprocess_server() -> Iterator[SubprocessServer]:
    """One fresh uvicorn production server for the RSS-growth node."""
    server = spawn_baseline_server()
    try:
        wait_until_ready(server.port)
        yield server
    finally:
        stop_server(server)
