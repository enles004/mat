"""Warm normalization budget evidence (MAT test plan, Task 5).

Measures the production catalog's compiled matcher + overlap-resolving
normalizer on a representative Vietnamese request string: 1,000 warmup
calls, then 10,000 measured calls with ``time.perf_counter_ns``. The gate
is warm p95 < 1.0 ms on the reference runner (Intel i5-14600KF, 30 GiB
RAM); the full distribution and runner metadata are attached to the test
report and printed for the evidence ledger.

Marker ``performance`` keeps this out of the fast profile; run it with
``-m performance``.
"""

import math
import os
import platform
import sys
import time
from pathlib import Path

import pytest

from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer

pytestmark = pytest.mark.performance

PROD_CATALOG = Path("configs/normalization.yaml")
WARMUP_CALLS = 1_000
MEASURED_CALLS = 10_000
P95_BUDGET_MS = 1.0

# Representative request text: exercises the hyundai brand rule, the
# negation rule, and both teencode rules in one realistic sentence.
SAMPLE_TEXT = "Xe H. cx ko dc, hãng nói cx ko sao, dùng ổn định, ko dc bấm giờ"


def _percentile_ms(sorted_ms: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an ascending sample."""
    rank = max(1, math.ceil(fraction * len(sorted_ms)))
    return sorted_ms[rank - 1]


def test_warm_normalize_p95_stays_under_one_millisecond(
    request: pytest.FixtureRequest,
) -> None:
    loaded = CatalogLoader().load(PROD_CATALOG)
    normalizer = TextNormalizer(RegexMatcher.from_catalog(loaded.catalog))

    sanity = normalizer.normalize(SAMPLE_TEXT)
    assert sanity.normalized_text != SAMPLE_TEXT, "sample must exercise the catalog"
    assert not sanity.warnings

    for _ in range(WARMUP_CALLS):
        normalizer.normalize(SAMPLE_TEXT)

    durations_ms: list[float] = []
    for _ in range(MEASURED_CALLS):
        started = time.perf_counter_ns()
        normalizer.normalize(SAMPLE_TEXT)
        durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    ordered = sorted(durations_ms)
    p50 = _percentile_ms(ordered, 0.50)
    p90 = _percentile_ms(ordered, 0.90)
    p95 = _percentile_ms(ordered, 0.95)
    p99 = _percentile_ms(ordered, 0.99)
    maximum = ordered[-1]
    evidence = (
        f"normalization warm p95 evidence | catalog={loaded.checksum} "
        f"dictionary_version={loaded.catalog.dictionary_version} "
        f"rule_count={len(loaded.catalog.rules)} "
        f"platform={platform.platform()} python={sys.version.split()[0]} "
        f"cpus={os.cpu_count()} warmup={WARMUP_CALLS} measured={MEASURED_CALLS} "
        f"p50={p50:.4f}ms p90={p90:.4f}ms p95={p95:.4f}ms "
        f"p99={p99:.4f}ms max={maximum:.4f}ms budget_p95<{P95_BUDGET_MS}ms"
    )
    request.node.add_report_section("call", "normalization-budget", evidence)
    print(evidence)

    assert p95 < P95_BUDGET_MS, evidence
