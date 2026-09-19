# Verification — Task 13 final evidence bundle

Purpose: command-by-command evidence for the documentation, cards, and packaging
task. All commands were executed on 2026-09-17 (UTC timestamps below) from the
worktree `/home/faris/code/MAT/.worktrees/mat-sentiment-core`, git revision
`b14a3ca` (short SHA; branch `feature/mat-sentiment-core`), with the Task 13
documentation files present as uncommitted working-tree changes (the commands
under test do not depend on those files being committed). Exit codes were captured
directly from the shell.

## Step 1+2 — Documentation contract test (TDD RED → GREEN)

| Command | Timestamp (UTC) | Exit | Result |
| --- | --- | ---: | --- |
| `uv run --locked pytest tests/packaging/test_documented_commands.py -q` (against the pre-Task-13 `ai/README.md`) | 2026-09-17T07:32:55Z (reproduced; original RED observed immediately after writing the test, before any README edit) | 1 | RED: `AssertionError: assert 'mat-validate-data' in '# MAT AI\n\n…'` — `ai/README.md` lacked 5 of the 7 required literal commands |
| same command (after README update) | 2026-09-17T07:32:55Z | 0 | GREEN: `1 passed in 0.00s` |

The RED was captured by temporarily restoring the committed HEAD version of
`ai/README.md` (via `git show HEAD:ai/README.md`), running the test once, and
immediately restoring the Task 13 README — no repository state was lost.

Verbatim RED failure block:

```text
FAILED tests/packaging/test_documented_commands.py::test_ai_readme_contains_required_reproduction_commands
1 failed in 0.01s
```

(with `AssertionError: assert 'mat-validate-data' in '# MAT AI\n\nInstallable
Python package for Vietnamese automotive sentiment training, evaluation, and
inference. This ...d mypy\n```\n\nThe final service packaging path is Docker-only;
local commands are for development and verification.\n'`)

## Step 5 — Local verification

| Command | Timestamp (UTC) | Exit | Result |
| --- | --- | ---: | --- |
| `uv sync --locked` | 2026-09-17T07:26:13Z | 0 | environment synced to `uv.lock`; note: plain sync (no extras) uninstalls the `transformer` extra, and 4 test modules import it (see note below) |
| `uv sync --locked --extra transformer` | 2026-09-17T07:27:34Z | 0 | extra reinstalled — required for the full suite because `src/mat_ai/modeling/calibration.py` imports `torch` at module level (documented in `ai/README.md`) |
| `uv run --locked pytest -q` | 2026-09-17T07:28:21Z | 0 | `100 passed, 2 warnings in 3.86s` (99 pre-existing + 1 new documentation test) |
| `uv run --locked ruff check src tests` | 2026-09-17T07:28:35Z | 0 | `All checks passed!` |
| `uv run --locked mypy` | 2026-09-17T07:28:36Z | 0 | `Success: no issues found in 31 source files` |
| `uv run --locked mat-validate-data --input data/dataset.csv --split-manifest data/split_manifest.json` | 2026-09-17T07:28:49Z | 0 | `{"class_counts": {"negative": 300, "neutral": 300, "positive": 300}, "errors": [], "noise_ratio": 0.36666666666666664, "row_count": 900, "warnings": []}` |
| `uv run --locked python -m mat_ai.modeling.registry verify artifacts/baseline` | 2026-09-17T07:28:49Z | 0 | `{"artifact": "artifacts/baseline", "backend": "linear", "model_name": "tfidf-word-char-logreg", "model_version": "1.0.0", "payload_checksum": "sha256:c89ecce613216be75e58b69b04b15fb6d05cd5cd72a9be0d5ac1aa37849e8af6"}` — matches the manifest checksum recorded in `reports/model-selection.md` |

Hygiene gates:

| Command | Timestamp (UTC) | Exit | Result |
| --- | --- | ---: | --- |
| `uv lock --check` | 2026-09-17T07:28:59Z | 0 | `Resolved 107 packages in 1ms` — lockfile up to date |
| `sha256sum data/dataset.csv data/split_manifest.json` | 2026-09-17T07:28:59Z | 0 | matches the frozen values (bottom of this file) |
| `sha256sum data/challenge_set.csv` | 2026-09-17T07:33:34Z | 0 | `e668df0e…36170d4` — matches `source_checksums.challenge_set` in `reports/baseline-evaluation.json` |

Per the plan and brief, `mat-train-baseline` and `mat-evaluate` were **not**
re-run in this task: both are deterministic re-creation commands documented in
`ai/README.md` for evaluators, and re-running them would only rewrite identical
outputs (plan Step 5 does not include them).

## Step 6 — Clean Docker verification

All commands from `infra/`. The build used `--no-cache`: every `RUN` layer was
re-executed (`uv sync --frozen --no-dev` took 34.5 s; nothing except the two base
image references was cache-hit).

| Command | Timestamp (UTC) | Exit | Result |
| --- | --- | ---: | --- |
| `docker compose build --no-cache ai` | 2026-09-17T07:29:26Z → 07:30:21Z | 0 | clean build succeeded (~55 s); image `infra-ai:latest` written |
| `docker compose up -d --wait ai` | 2026-09-17T07:30:58Z → 07:31:04Z | 0 | container `infra-ai-1` reached `Healthy` (`--wait` adaptation approved in Task 12; avoids racing curl against startup) |
| `docker compose ps` | 2026-09-17T07:31:04Z | 0 | `infra-ai-1 … Up 5 seconds (healthy)` with `0.0.0.0:8000->8000/tcp` |
| `curl --fail http://127.0.0.1:8000/livez` | 2026-09-17T07:31:09Z | 0 | `{"status":"ok"}` |
| `curl --fail http://127.0.0.1:8000/readyz` | 2026-09-17T07:31:09Z | 0 | `{"backend":"linear","degraded":false,"version":"1.0.0"}` |
| `curl --fail -H 'Content-Type: application/json' -d '{"text":"Xe nhà Hyundai chạy ổn nhưng hơi hao xăng"}' http://127.0.0.1:8000/predict` | 2026-09-17T07:31:18Z | 0 | body below — all required fields present: `label`, `confidence`, three-class `scores`, `model`, `request_id` |
| `docker compose down` | 2026-09-17T07:31:27Z | 0 | container and network removed; `docker compose ps` empty |

Predict response body (verbatim):

```json
{"label":"positive","confidence":0.8180027010619891,"scores":{"negative":0.09414290497489668,"neutral":0.08785439396311409,"positive":0.8180027010619891},"uncertain":false,"model":{"backend":"linear","version":"1.0.0","degraded":false},"request_id":"req_4f68b1a4cad647efb40fd501fba480cf"}
```

### Base image digests resolved by this build (M1 drift-check reference)

The Dockerfile pins mutable tags. During this verification they resolved to:

| Tag | Resolved digest |
| --- | --- |
| `python:3.12-slim` | `sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a` |
| `ghcr.io/astral-sh/uv:0.11.16` | `sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d` |

Deferred M1 (from Task 12): Python dependency reproducibility is guaranteed by
`uv.lock`, but these base tags can be re-pointed by their registries; a rebuild
for release should diff the resolved digests against this table (also recorded in
`reports/model_card.md` limitations).

## Limitations of this verification

- The Docker base images above were resolved from local registry metadata on this
  host; a different host/date may resolve the same mutable tags to newer digests.
- GPU measurements (transformer candidate environment) were recorded during Task 8
  and are quoted in `reports/model_card.md`; they were not re-measured here.
- Local commands ran with the Task 13 file changes uncommitted in the working
  tree; the commit that follows contains exactly those files (see `git status`
  in the task report).

## Known provenance notes (final review, 2026-09-17)

- `reports/baseline-evaluation.{json,md}` were generated by the Task 7 code and
  are kept byte-frozen as reviewed evidence. Regenerating them with the current
  code adds two descriptive fields (`candidate_type`, `challenge_inference`,
  introduced for the Task 8 transformer candidate) while every metric stays
  digit-identical — proven by an independent regeneration diff during the final
  whole-project review. The committed files are intentionally not regenerated.

## Final gate re-run (immediately before commit)

All gates re-executed after the last documentation edit, at
2026-09-17T07:33:56Z, revision `b14a3ca` + working-tree Task 13 files:

| Command | Exit | Result |
| --- | ---: | --- |
| `uv run --locked pytest -q` | 0 | `100 passed, 2 warnings in 3.97s` |
| `uv run --locked ruff check src tests` | 0 | `All checks passed!` |
| `uv run --locked mypy` | 0 | `Success: no issues found in 31 source files` |
| `uv lock --check` | 0 | lockfile up to date |
| `git status --short` (worktree root) | 0 | only the 8 Task 13 files (2 modified, 6 new); `Makefile` unchanged — no new target was promised |
| `sha256sum data/dataset.csv data/split_manifest.json` (from `ai/`) | 0 | both match the frozen values below |

## Post-review polish round (final-review findings M2/M1 + triaged minors)

After the independent whole-project review returned APPROVE with 2 Minor
findings, one polish commit was applied on top of `3abd5b0`: the clean-clone
`uv sync` line now installs the `transformer` extra up front (README + Makefile
`sync` target) so the literal command sequence no longer fails test collection;
the Task 6 fold-disjointness `assert` became a runtime `ValueError` (survives
`python -O`, with a new test proving the rejection); a regression test now
asserts the checked-in `split_manifest.json` equals a fresh deterministic
rebuild; and this file gained the provenance note above. Gates re-run at
revision `3abd5b0` + working-tree polish files (2026-09-17, UTC; local gates
~12:31Z, Docker cycle 12:32:28Z → 12:33:09Z):

| Command | Exit | Result |
| --- | ---: | --- |
| `uv run --locked pytest -q` | 0 | `102 passed, 2 warnings in 4.03s` (100 + 2 new) |
| `uv run --locked ruff check src tests` | 0 | `All checks passed!` |
| `uv run --locked mypy` | 0 | `Success: no issues found in 31 source files` |
| `uv lock --check` | 0 | lockfile up to date |
| `uv run --locked pytest tests/packaging/test_documented_commands.py -q` | 0 | `1 passed` (README still carries all 7 literal command strings) |
| `sha256sum data/dataset.csv data/split_manifest.json` | 0 | both match the frozen values below |
| `cd ../infra && docker compose build --no-cache ai` | 0 | clean rebuild succeeded; container image unchanged in content |
| `docker compose up -d --wait ai` | 0 | container reached `Healthy` |
| `curl --fail http://127.0.0.1:8000/livez` | 0 | `{"status":"ok"}` |
| `curl --fail http://127.0.0.1:8000/readyz` | 0 | `{"backend":"linear","degraded":false,"version":"1.0.0"}` |
| `curl --fail -H 'Content-Type: application/json' -d '{"text":"Xe nhà Hyundai chạy ổn nhưng hơi hao xăng"}' http://127.0.0.1:8000/predict` | 0 | `positive 0.8180027010619891` — digit-identical to the Step 6 body |
| `docker compose down` | 0 | container and network removed |

## Frozen checksums

| File | SHA-256 |
| --- | --- |
| `data/dataset.csv` | `2a645d1605d6d7a0e1b092f15eba2c0b75a3a0006af0ee23fc1aceff0f374501` |
| `data/split_manifest.json` | `7c89e0df8838b2662141236994ab71dfef5412fc504cdea9e613307988fd783f` |
