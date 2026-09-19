"""Release batch/throughput/memory budgets (MAT test plan, Task 8).

Three nodes over the frozen 1,028-input corpus (54 showcase + 74 challenge +
900 dataset, loaded through the ``verify_corpus_equivalence`` loader):

1. RSS growth — repeated full 1,028-input batches against a fresh uvicorn
   production subprocess; ``VmRSS`` is read from ``/proc`` after a short
   stabilization pause and the worst stabilized growth is gated.
2. Sequential throughput — every corpus input scored once, sequentially,
   through the in-process production app; total wall time and req/s gated.
3. Bounded concurrent burst — a fixed 8-worker x 50-request burst through the
   in-process production app; every response is fully validated (canonical
   label, finite scores summing to one, distinct well-formed request IDs) and
   throughput gated.

Budgets are versioned numbers with provenance: the median of three
independent measurement rounds on the Task 0 reference runner (Intel
i5-14600KF, 20 threads, 30 Gi RAM), multiplied by 1.20, recorded in
``ai/reports/test-performance-baseline.md``. No budget below is aspirational.
"""

import asyncio
import math
import platform
import time
from typing import Final

import httpx
import pytest
from fastapi import FastAPI

from src.domain.entities import SentimentLabel
from tests.performance.conftest import (
    SubprocessServer,
    read_vmrss_kb,
)

pytestmark = pytest.mark.performance

MEASURED_BATCHES = 2
RSS_SETTLE_S: Final[float] = 1.0
BURST_WORKERS = 8
BURST_REQUESTS_PER_WORKER = 50
VALID_LABELS = {label.value for label in SentimentLabel}
REQUEST_ID_PATTERN_LENGTH = len("req_") + 32

# Budget provenance: ai/reports/test-performance-baseline.md (2026-09-18).
# Reference runner: Intel i5-14600KF, 20 threads, 30 Gi RAM, Linux
# 7.0.0-31-generic, CPython 3.12.13. Each budget is the median of three
# independent measurement rounds multiplied by 1.20 (floored), never a
# retuned number. Median round values: RSS worst growth 532 kB (budget as
# kB), sequential total 5.3542 s over 1,028 inputs, burst 191.3 req/s
# (floor = median req/s / 1.20, equivalent to wall-time x1.20).
RSS_GROWTH_BUDGET_KB = 638
SEQUENTIAL_TOTAL_BUDGET_S = 6.4250
BURST_MIN_RPS = 159.4


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an ascending sample."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _score_full_corpus(client: httpx.Client, payloads: list[dict[str, str]]) -> None:
    """One sequential full-corpus pass; every response must be a valid 200."""
    for payload in payloads:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, (payload, response.status_code)


def test_rss_growth_after_repeated_full_batches_stays_within_budget(
    baseline_subprocess_server: SubprocessServer,
    corpus_inputs: list[tuple[str, str, str]],
    request: pytest.FixtureRequest,
) -> None:
    """Repeated full-corpus batches against one server; stabilized VmRSS gated."""
    server = baseline_subprocess_server
    pid = server.process.pid
    payloads = [{"text": text} for _, _, text in corpus_inputs]
    with httpx.Client(base_url=server.base_url, timeout=30.0) as client:
        _score_full_corpus(client, payloads)  # warmup batch: allocator settles
        time.sleep(RSS_SETTLE_S)
        settled_base_kb = read_vmrss_kb(pid)
        growths_kb: list[int] = []
        rss_trace: list[int] = [settled_base_kb]
        for _ in range(MEASURED_BATCHES):
            _score_full_corpus(client, payloads)
            time.sleep(RSS_SETTLE_S)
            settled_kb = read_vmrss_kb(pid)
            rss_trace.append(settled_kb)
            growths_kb.append(settled_kb - settled_base_kb)
    worst_growth_kb = max(growths_kb)
    evidence = (
        f"rss growth | platform={platform.platform()} pid={pid} "
        f"warmup_batches=1 measured_batches={MEASURED_BATCHES} "
        f"inputs_per_batch={len(payloads)} settle_s={RSS_SETTLE_S} "
        f"settled_base_kb={settled_base_kb} rss_trace_kb={rss_trace} "
        f"worst_growth_kb={worst_growth_kb} budget<={RSS_GROWTH_BUDGET_KB}kB"
    )
    request.node.add_report_section("call", "rss-growth", evidence)
    print(evidence)
    assert worst_growth_kb <= RSS_GROWTH_BUDGET_KB, evidence


def test_sequential_full_corpus_throughput_meets_budget(
    baseline_app: FastAPI,
    corpus_inputs: list[tuple[str, str, str]],
    request: pytest.FixtureRequest,
) -> None:
    """All 1,028 corpus inputs scored sequentially; total wall time gated."""
    payloads = [{"text": text} for _, _, text in corpus_inputs]

    async def run() -> float:
        transport = httpx.ASGITransport(app=baseline_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            started = time.perf_counter_ns()
            for payload in payloads:
                response = await client.post("/predict", json=payload)
                assert response.status_code == 200, (payload, response.status_code)
            return (time.perf_counter_ns() - started) / 1_000_000_000

    total_s = asyncio.run(run())
    requests_per_s = len(payloads) / total_s
    evidence = (
        f"sequential corpus throughput | platform={platform.platform()} "
        f"inputs={len(payloads)} total_s={total_s:.4f} req_per_s={requests_per_s:.1f} "
        f"budget_total_s<={SEQUENTIAL_TOTAL_BUDGET_S:.4f}"
    )
    request.node.add_report_section("call", "sequential-throughput", evidence)
    print(evidence)
    assert total_s <= SEQUENTIAL_TOTAL_BUDGET_S, evidence


def test_bounded_concurrent_burst_returns_valid_isolated_responses(
    baseline_app: FastAPI,
    corpus_inputs: list[tuple[str, str, str]],
    request: pytest.FixtureRequest,
) -> None:
    """8 workers x 50 requests; every response fully validated, throughput gated."""
    total_requests = BURST_WORKERS * BURST_REQUESTS_PER_WORKER
    payloads = [
        {"text": corpus_inputs[index % len(corpus_inputs)][2]} for index in range(total_requests)
    ]

    async def run() -> tuple[list[httpx.Response], float]:
        transport = httpx.ASGITransport(app=baseline_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:

            async def worker(worker_index: int) -> list[httpx.Response]:
                responses: list[httpx.Response] = []
                offset = worker_index * BURST_REQUESTS_PER_WORKER
                for step in range(BURST_REQUESTS_PER_WORKER):
                    response = await client.post("/predict", json=payloads[offset + step])
                    responses.append(response)
                return responses

            started = time.perf_counter_ns()
            batches = await asyncio.gather(*(worker(i) for i in range(BURST_WORKERS)))
            elapsed_s = (time.perf_counter_ns() - started) / 1_000_000_000
            return [response for batch in batches for response in batch], elapsed_s

    responses, elapsed_s = asyncio.run(run())
    assert len(responses) == total_requests
    request_ids: list[str] = []
    for response in responses:
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["label"] in VALID_LABELS, body
        scores = body["scores"]
        assert set(scores) == VALID_LABELS, body
        assert all(math.isfinite(value) for value in scores.values()), body
        assert abs(sum(scores.values()) - 1.0) <= 1e-6, body
        request_id = body["request_id"]
        assert len(request_id) == REQUEST_ID_PATTERN_LENGTH and request_id.startswith("req_"), body
        request_ids.append(request_id)
    assert len(set(request_ids)) == total_requests, "request IDs must be unique per burst"

    requests_per_s = total_requests / elapsed_s
    evidence = (
        f"concurrent burst | platform={platform.platform()} "
        f"workers={BURST_WORKERS} requests_per_worker={BURST_REQUESTS_PER_WORKER} "
        f"total={total_requests} elapsed_s={elapsed_s:.4f} req_per_s={requests_per_s:.1f} "
        f"budget_min_rps>={BURST_MIN_RPS:.1f}"
    )
    request.node.add_report_section("call", "concurrent-burst", evidence)
    print(evidence)
    assert requests_per_s >= BURST_MIN_RPS, evidence
