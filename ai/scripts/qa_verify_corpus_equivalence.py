#!/usr/bin/env python3
"""Corpus equivalence verifier — proof tool for clean-architecture Task 9.

Runs the frozen 1,014-text corpus (54 showcase + 60 challenge + 900 dataset)
against two live /predict servers and proves zero semantic difference.

Canonicalization: drop request_id/timing-like keys only; everything else
(label, confidence, scores, uncertainty, backend, version, degraded, ...)
must be identical, including float formatting (repr-compared after
json round-trip with sort_keys and no reformatting of numbers).

Data directory resolution: $MAT_AI_DATA, else the sibling ``data/`` of this
script's parent (i.e. ``ai/data`` in a checkout or ``/app/data`` in a wheel).

Usage: python scripts/qa_verify_corpus_equivalence.py REF_URL CAND_URL
Prints: per-source counts, mismatch details, canonical SHA-256 per side.
Exit 0 = identical; exit 1 = mismatch.
"""
import csv
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

DROP_RE = re.compile(r"request|timing|elapsed|latency|duration", re.I)


def _data_dir() -> str:
    env = os.environ.get("MAT_AI_DATA")
    if env:
        return env
    return str(Path(__file__).resolve().parents[1] / "data")


def load_corpus() -> dict[str, list[tuple[str, str]]]:
    data = _data_dir()
    showcase = json.load(open(f"{data}/vietnamese_demo_cases.json"))["cases"]
    showcase = [(c["id"], c["text"]) for c in showcase]
    challenge = []
    with open(f"{data}/challenge_set.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            challenge.append((row["id"], row["text"]))
    dataset = []
    with open(f"{data}/dataset.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset.append((row["id"], row["raw_text"]))
    return {"showcase": showcase, "challenge": challenge, "dataset": dataset}


def canonicalize(resp: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in resp.items() if not DROP_RE.search(k)}


def predict(base: str, text: str) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/predict",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        payload: dict[str, Any] = json.load(r)
        return payload


def main() -> None:
    ref, cand = sys.argv[1], sys.argv[2]
    corpus = load_corpus()
    total = mismatches = 0
    dumps: dict[str, list[str]] = {"ref": [], "cand": []}
    for source, items in corpus.items():
        src_mis = 0
        for cid, text in items:
            r = canonicalize(predict(ref, text))
            c = canonicalize(predict(cand, text))
            jr = json.dumps(r, sort_keys=True, ensure_ascii=False)
            jc = json.dumps(c, sort_keys=True, ensure_ascii=False)
            dumps["ref"].append(jr)
            dumps["cand"].append(jc)
            total += 1
            if jr != jc:
                mismatches += 1
                src_mis += 1
                if src_mis <= 3:
                    print(f"MISMATCH [{source}/{cid}] {text[:60]!r}")
                    print(f"  REF : {jr[:300]}")
                    print(f"  CAND: {jc[:300]}")
        print(f"[{source}] {len(items)} texts, {src_mis} mismatches")
    h_ref = hashlib.sha256("\n".join(dumps["ref"]).encode()).hexdigest()
    h_cand = hashlib.sha256("\n".join(dumps["cand"]).encode()).hexdigest()
    print(f"TOTAL {total} texts, {mismatches} mismatches")
    print(f"REF  sha256 {h_ref}")
    print(f"CAND sha256 {h_cand}")
    print("EQUIVALENT" if (mismatches == 0 and h_ref == h_cand) else "DIFFERENT")
    sys.exit(0 if (mismatches == 0 and h_ref == h_cand) else 1)


if __name__ == "__main__":
    main()
