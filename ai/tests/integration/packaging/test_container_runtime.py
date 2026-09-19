# ai/tests/integration/packaging/test_container_runtime.py
"""Real container runtime gates for the packaged MAT service (Plan 2 Task 6).

Drives an actual Docker Compose deployment of the built image through four
serial gates: dedicated non-root UID, serving under a read-only root
filesystem with a tmpfs ``/tmp``, a stop/start cycle returning to readiness
with clean teardown, and an RFC 9457 malformed-request answer over the
published port.

Serial by design: every node reuses one uniquely named Compose project,
gets a bounded readiness poll, and is guaranteed a
``down --volumes --remove-orphans`` teardown. Never run this module under a
parallel test runner; the ``release`` marker keeps it out of the fast
profile (run with ``-m release``).
"""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPOSE_FILE = REPO_ROOT / "infra" / "docker-compose.yml"
EXPECTED_UID = 10001
READY_DEADLINE_S = 180.0

pytestmark = pytest.mark.release


def _compose(project: str, *args: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--project-name", project, "-f", str(COMPOSE_FILE), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _override_file(tmp_path: Path, port: int, *, read_only: bool) -> Path:
    """A Compose override that repoints the published port to a free host
    port (``!override`` replaces the fixed ``8000:8000`` mapping) and, for
    the read-only gate, declares a read-only root filesystem plus a tmpfs
    ``/tmp``."""
    lines = [
        "services:",
        "  ai:",
        "    ports: !override:",
        f'      - "127.0.0.1:{port}:8000"',
    ]
    if read_only:
        lines += ["    read_only: true", "    tmpfs:", "      - /tmp"]
    path = tmp_path / "compose-override.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _wait_until_ready(base_url: str) -> None:
    """Bounded readiness polling against the published port."""
    deadline = time.monotonic() + READY_DEADLINE_S
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health-check", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # connection refused until uvicorn binds
            last = exc
        time.sleep(1.0)
    pytest.fail(f"/health-check never became ready within {READY_DEADLINE_S:.0f}s: {last}")


class ComposeService:
    """One serial deployment of the compose project under test."""

    def __init__(self, project: str) -> None:
        self.project = project
        self.base_url = ""
        self.started = False

    def up(self, override: Path | None = None) -> None:
        command = ["docker", "compose", "--project-name", self.project, "-f", str(COMPOSE_FILE)]
        if override is not None:
            command += ["-f", str(override)]
        output = ""
        for attempt in range(2):
            if attempt:
                # One retry absorbs a cold-start unhealthy verdict right after
                # the uncached build (disk contention can outlive the 3s
                # healthcheck timeout); a second consecutive verdict fails.
                _compose(self.project, "down", "--volumes", "--remove-orphans", "--timeout", "60")
            result = subprocess.run(
                [*command, "up", "-d", "--wait", "--wait-timeout", str(int(READY_DEADLINE_S))],
                capture_output=True,
                text=True,
                timeout=READY_DEADLINE_S + 120.0,
                check=False,
            )
            if result.returncode == 0:
                self.started = True
                return
            output = result.stderr or result.stdout
        pytest.fail(f"compose up never became healthy: {output}")

    def plain(self, *args: str) -> subprocess.CompletedProcess[str]:
        return _compose(self.project, *args)

    def exec(self, *command: str) -> subprocess.CompletedProcess[str]:
        return self.plain("exec", "-T", "ai", *command)

    def down(self) -> subprocess.CompletedProcess[str]:
        self.started = False
        return self.plain("down", "--volumes", "--remove-orphans", "--timeout", "60")


@pytest.fixture(scope="module")
def project() -> Iterator[str]:
    """Build the image once, uncached, under a unique project name; the
    safety-net teardown guarantees no scoped resources outlive the module."""
    name = f"mat-test-{uuid.uuid4().hex[:12]}"
    build = _compose(name, "build", "--no-cache", timeout=2400.0)
    assert build.returncode == 0, build.stderr[-4000:] or build.stdout[-4000:]
    yield name
    _compose(name, "down", "--volumes", "--remove-orphans")


@pytest.fixture
def service(project: str) -> Iterator[ComposeService]:
    runtime = ComposeService(project)
    yield runtime
    if runtime.started:
        runtime.down()


def _predict(base_url: str, body: bytes, timeout: float = 30.0):
    request = urllib.request.Request(
        f"{base_url}/predict",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        return error  # HTTPError is a response-like carrying status, headers, body


def test_container_runs_as_dedicated_non_root_user(
    service: ComposeService,
    tmp_path: Path,
) -> None:
    override = _override_file(tmp_path, _free_port(), read_only=False)
    service.up(override)

    result = service.exec("id", "-u")
    assert result.returncode == 0, result.stderr
    uid = result.stdout.strip()
    assert uid != "0", "the serving container must not run as root"
    assert uid == str(EXPECTED_UID), uid


def test_read_only_root_filesystem_with_tmp_tmpfs_still_serves(
    service: ComposeService,
    tmp_path: Path,
) -> None:
    port = _free_port()
    override = _override_file(tmp_path, port, read_only=True)
    service.up(override)
    service.base_url = f"http://127.0.0.1:{port}"
    _wait_until_ready(service.base_url)

    body = json.dumps({"text": "Xe này rất tốt"}).encode()
    with _predict(service.base_url, body) as response:
        assert response.status == 200
        payload = json.loads(response.read())
    assert payload["label"] in {"negative", "neutral", "positive"}
    assert payload["request_id"]


def test_compose_stop_start_cycle_returns_to_ready_and_teardown_is_clean(
    service: ComposeService,
    tmp_path: Path,
) -> None:
    port = _free_port()
    service.up(_override_file(tmp_path, port, read_only=False))
    service.base_url = f"http://127.0.0.1:{port}"
    _wait_until_ready(service.base_url)

    stopped = service.plain("stop")
    assert stopped.returncode == 0, stopped.stderr or stopped.stdout

    started = service.plain("start")
    assert started.returncode == 0, started.stderr or started.stdout
    _wait_until_ready(service.base_url)
    body = json.dumps({"text": "Xe này rất tốt"}).encode()
    with _predict(service.base_url, body) as response:
        assert response.status == 200

    teardown = service.down()
    assert teardown.returncode == 0, teardown.stderr or teardown.stdout

    containers = service.plain("ps", "-a", "--format", "json")
    assert containers.returncode == 0, containers.stderr
    assert not [line for line in containers.stdout.splitlines() if line.strip()], containers.stdout

    volumes = subprocess.run(
        [
            "docker",
            "volume",
            "ls",
            "--format",
            "{{.Name}}",
            "--filter",
            f"label=com.docker.compose.project={service.project}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert volumes.returncode == 0, volumes.stderr
    assert volumes.stdout.strip() == "", volumes.stdout


def test_container_answers_malformed_json_with_rfc9457_problem(
    service: ComposeService,
    tmp_path: Path,
) -> None:
    port = _free_port()
    override = _override_file(tmp_path, port, read_only=False)
    service.up(override)
    service.base_url = f"http://127.0.0.1:{port}"
    _wait_until_ready(service.base_url)

    with _predict(service.base_url, b"{not json") as response:
        assert response.status == 400
        assert response.headers.get_content_type() == "application/problem+json"
        body = json.loads(response.read())
    assert body["code"] == "MALFORMED_JSON"
    assert body["status"] == 400
    assert body["request_id"]
    assert body["type"].startswith("https://")
