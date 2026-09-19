"""Release API performance budgets (MAT test plan, Task 8).

Two nodes against the production baseline backend:

1. Warm ``/predict`` latency — 100 warmups then 1,000 sequential requests
   through the in-process production app, timed with ``time.perf_counter_ns``
   and gated on nearest-rank p50/p95/p99.
2. Fresh-process startup/readiness — 20 spawned uvicorn ``main:app`` servers;
   each sample is the wall time from process spawn to the first ``/health-check``
   200. The round metric is the median of the 20 samples.

Budgets are versioned numbers with provenance: the median of three
independent measurement rounds on the Task 0 reference runner (Intel
i5-14600KF, 20 threads, 30 Gi RAM), multiplied by 1.20, recorded in
``ai/reports/test-performance-baseline.md``. No budget below is aspirational.
"""

import asyncio
import math
import os
import platform
import statistics
import sys
import time

import httpx
import pytest
from fastapi import FastAPI

from tests.performance.conftest import spawn_baseline_server, stop_server, wait_until_ready

pytestmark = pytest.mark.performance

WARMUP_CALLS = 100
MEASURED_CALLS = 1_000
STARTUP_SAMPLES = 20

# Budget provenance: ai/reports/test-performance-baseline.md (2026-09-18).
# Reference runner: Intel i5-14600KF, 20 threads, 30 Gi RAM, Linux
# 7.0.0-31-generic, CPython 3.12.13. Each budget is the median of three
# independent measurement rounds multiplied by 1.20 (floored at 4 decimals),
# never a retuned number. Median round values: warm p50 5.3870 ms, p95
# 5.5319 ms, p99 5.6475 ms; startup median 0.7588 s.
WARM_P50_BUDGET_MS = 6.4644
WARM_P95_BUDGET_MS = 6.6382
WARM_P99_BUDGET_MS = 6.7770
STARTUP_MEDIAN_BUDGET_S = 0.9105


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an ascending sample."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def test_warm_predict_latency_percentiles_within_budget(
    baseline_app: FastAPI,
    corpus_inputs: list[tuple[str, str, str]],
    request: pytest.FixtureRequest,
) -> None:
    """100 warmups, then 1,000 sequential /predict samples, p50/p95/p99 gated."""
    sample_text = corpus_inputs[0][2]

    async def measure() -> list[float]:
        transport = httpx.ASGITransport(app=baseline_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            for _ in range(WARMUP_CALLS):
                warm = await client.post("/predict", json={"text": sample_text})
                assert warm.status_code == 200, warm.text
            durations_ms: list[float] = []
            for _ in range(MEASURED_CALLS):
                started = time.perf_counter_ns()
                response = await client.post("/predict", json={"text": sample_text})
                durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                assert response.status_code == 200, response.text
            return durations_ms

    ordered = sorted(asyncio.run(measure()))
    p50 = _percentile(ordered, 0.50)
    p95 = _percentile(ordered, 0.95)
    p99 = _percentile(ordered, 0.99)
    evidence = (
        f"warm api latency | platform={platform.platform()} "
        f"python={sys.version.split()[0]} cpus={os.cpu_count()} "
        f"warmup={WARMUP_CALLS} measured={MEASURED_CALLS} "
        f"p50={p50:.4f}ms p95={p95:.4f}ms p99={p99:.4f}ms "
        f"budget_p50<{WARM_P50_BUDGET_MS:.4f}ms p95<{WARM_P95_BUDGET_MS:.4f}ms "
        f"p99<{WARM_P99_BUDGET_MS:.4f}ms"
    )
    request.node.add_report_section("call", "warm-api-latency", evidence)
    print(evidence)
    assert p50 < WARM_P50_BUDGET_MS, evidence
    assert p95 < WARM_P95_BUDGET_MS, evidence
    assert p99 < WARM_P99_BUDGET_MS, evidence


def test_fresh_process_startup_reaches_ready_within_budget(
    request: pytest.FixtureRequest,
) -> None:
    """20 fresh uvicorn processes; median spawn-to-/health-check-200 time is gated."""
    samples_s: list[float] = []
    for _ in range(STARTUP_SAMPLES):
        started = time.perf_counter_ns()
        server = spawn_baseline_server()
        try:
            samples_s.append(wait_until_ready(server.port, started_ns=started))
        finally:
            stop_server(server)
    ordered = sorted(samples_s)
    median_s = statistics.median(ordered)
    evidence = (
        f"startup readiness | platform={platform.platform()} samples={STARTUP_SAMPLES} "
        f"min={ordered[0]:.4f}s median={median_s:.4f}s "
        f"p95={_percentile(ordered, 0.95):.4f}s max={ordered[-1]:.4f}s "
        f"budget_median<={STARTUP_MEDIAN_BUDGET_S:.4f}s"
    )
    request.node.add_report_section("call", "startup-readiness", evidence)
    print(evidence)
    assert median_s <= STARTUP_MEDIAN_BUDGET_S, evidence
