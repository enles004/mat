# Model card — MAT Vietnamese automotive sentiment champion

## Model identity

| Field | Value | Evidence |
| --- | --- | --- |
| Champion | `baseline` (sigmoid-calibrated linear pipeline) | `reports/model-selection.md` |
| Model name / version | `tfidf-word-char-logreg` 1.0.0, backend `linear` | `artifacts/baseline/manifest.json` |
| Exported artifact | `artifacts/baseline` (`model.joblib` + `manifest.json`) | manifest |
| Payload checksum | `sha256:c89ecce613216be75e58b69b04b15fb6d05cd5cd72a9be0d5ac1aa37849e8af6` | manifest |
| Training git revision | `ab04643e8a0fb7e7ad353cd175661bc1daed35fe` | manifest |
| License | `internal-use-only` (trained in this repository on reviewed synthetic data; no external terms) | manifest, `reports/model-selection.md` |
| Label order (fixed) | `negative` → `neutral` → `positive` | manifest, `configs/models.yaml` |
| Max input tokens | not applicable (linear backend applies no tokenizer limit) | manifest |

## Preprocessing

- Input feature view: **raw** (`raw_text`), the selected view after the raw vs
  normalized comparison (`selected_view: raw` in `reports/baseline-evaluation.json`;
  `chosen_view: raw` in `runs/baseline/cv_metrics.json`).
- Features: TF-IDF word unigrams–bigrams + `char_wb` TF-IDF 3–5 grams,
  `min_df=2`, `sublinear_tf=true` (preprocessor
  `tfidf-word-bigram-char_wb-trigram (raw view)` 1.0.0, config checksum
  `sha256:145266e2541ed68e01261d5ae8cd92f9d23302f57fd4dd36f7f1ccc8097852c7`;
  hyperparameters `classifier_c=2.0`, `max_iter=2000`, seed 42 from
  `configs/models.yaml`).
- Classifier: logistic regression with `class_weight="balanced"`
  (`src/mat_ai/modeling/linear.py`, `C=2.0`, `max_iter=2000`, seeded).
- Normalization rules exist as a separate auditable view (versioned YAML) and are
  **not** used by the champion (raw view chosen).

## Training, CV, and frozen-test protocol

1. **Development set**: 720 rows / 120 canonical groups, 5 persisted
   StratifiedGroupKFold folds (24 validation groups per fold, no group overlap;
   verified in `data/split_manifest.json`).
2. **Out-of-fold (OOF) training**: per fold, a fresh pipeline is trained on 576 rows
   and predicts the 144 held-out rows; the union of OOF predictions is the 720-row
   `runs/baseline/cv_predictions.csv`.
3. **Calibration** (sigmoid): cross-fitted over persisted folds — for each held-out
   fold the calibrator is fitted only on the other four folds' OOF outputs; the final
   exported calibrator is fitted on all 720 OOF rows by ensembling the five persisted
   fold calibrators. No row is ever calibrated on itself
   (`reports/model-selection.md`).
4. **Selection gate**: sigmoid calibration was kept because proper scores improved
   (log loss −0.023830, Brier −0.001993) and macro-F1 dropped by at most the
   configured 0.01 (−0.002775).
5. **Frozen test (single run)**: the final pipeline was fitted on all 720 development
   rows and evaluated **once** on the 180 frozen-test rows (30 groups, 60 per label);
   no selection decision was revisited from these numbers
   (`reports/model-selection.md`).
6. **Behavioral challenge**: 60 cases in `data/challenge_set.csv` (MFT/INV/DIR),
   disjoint from fitting and aggregate metrics.

## Metrics

### OOF development metrics (720 rows, native probabilities)

Source: `reports/baseline-evaluation.json`.

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.994439 |
| Log loss | 0.073565 |
| Multiclass Brier | 0.015494 |

Per class (OOF): negative P 0.991736 / R 1.000000 / F1 0.995851;
neutral P 0.995798 / R 0.987500 / F1 0.991632;
positive P 0.995833 / R 0.995833 / F1 0.995833.
Confusion (rows = actual, order negative/neutral/positive):
[[240, 0, 0], [2, 237, 1], [0, 1, 239]].

Fold macro-F1 (144 rows each): 0.979008, 1.000000, 1.000000, 1.000000, 0.993055.

### Calibration effect (cross-fitted OOF)

Source: `reports/model-selection.md`.

| Metric | Native OOF | Cross-fitted calibrated OOF |
| --- | ---: | ---: |
| Log loss | 0.073565 | 0.049734 |
| Multiclass Brier | 0.015494 | 0.013500 |
| Macro-F1 | 0.994439 | 0.991664 |

### Frozen-test metrics (180 rows, single run)

Source: `reports/model-selection.md`.

| Metric | Value |
| --- | ---: |
| Macro-F1 | 1.000000 |
| Log loss | 0.034893 |
| Multiclass Brier | 0.004703 |

Per class: 1.000000 precision/recall/F1 for all three labels (60 support each);
confusion matrix is perfectly diagonal (60/60/60).

Frozen-test slice highlights (all macro-F1 1.000000): noise types
code_switching 12 / missing_diacritics 24 / repeated_character 10 rows;
aspects after_sales 42, comfort 42, design 24, engine 24, fuel 48 rows;
all six styles 30 rows each; easy/medium/hard 60 rows each. Full tables in
`reports/model-selection.md` and consolidated in `reports/evaluation.md`.

### Behavioral challenge (committed challenge set, development-only evidence)

Source: `reports/baseline-evaluation.json` (`challenge`).

- Pipeline evaluation: **23/60 passed (0.383333)** — by kind: MFT 8/20, INV 11/20,
  DIR 4/20; by phenomenon the weakest are negation 0/6, restrained_sarcasm 0/3,
  spacing 0/2 (full table in `reports/baseline-evaluation.json`).
- Exported-artifact re-check: **22/60 (0.366667)** when `artifacts/baseline` itself
  was re-run over the same set (`reports/model-selection.md`, behavioral re-check).

> **Read this honestly.** Every aggregate number above was measured on synthetic,
> human-reviewed, generated text whose families are related across splits. These are
> **not production accuracy estimates** for real Vietnamese automotive opinions. The
> behavioral challenge — the closest proxy to difficult real input — passed only
> 23/60 (38.33%). See `reports/error_analysis.md` for the failure inventory.

## Measured environment

Baseline (champion) measurements, from `reports/model-selection.md`:

| Measurement | Value |
| --- | ---: |
| Per-row predict latency mean | 5.181375 ms |
| Per-row predict latency p50 | 5.182021 ms |
| Per-row predict latency p95 | 5.360980 ms |
| Batch latency (180 rows) | 30.883424 ms |
| Platform | Linux-7.0.0-31-generic-x86_64-with-glibc2.43 |
| CPU | Intel(R) Core(TM) i5-14600KF |
| Python / scikit-learn / NumPy | 3.12.13 / 1.9.1 / 2.5.3 |

The transformer candidate was trained and evaluated on GPU: NVIDIA RTX 5060 Ti
(16 GiB, driver 595.84) with PyTorch 2.14.0+cu130 and Transformers 5.5.0
(recorded in the Task 8 execution report; process log is not a tracked repo file).

## Non-deployed candidate: transformer

| Field | Value | Evidence |
| --- | --- | --- |
| Candidate | `transformer` (BamiBERT fine-tune) | `runs/transformer/source.json` |
| Source revision (pinned) | `Qualcomm-AI-Research/BamiBERT` @ `57bc1340debbe4e348ec549047a763caebe4a977` | `runs/transformer/source.json` |
| `model.safetensors` SHA-256 | `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627` | `runs/transformer/source.json` |
| OOF macro-F1 / log loss / Brier | 0.997222 / 0.094769 / 0.017309 | `reports/transformer-evaluation.md` |
| Challenge pass rate | 40/60 (0.666667) | `reports/model-selection.md` |
| Eligibility | **Ineligible for deployment** — model card states "The model is intended for research and educational purposes." under the Qualcomm responsible-AI license; deployment would require documented approval of the exact terms | `runs/transformer/source.json`, `reports/transformer-license-audit.md` |
| Artifact export | None (metrics/predictions only; no payload exported) | `reports/model-selection.md` |

Ranking rule (macro-F1 first, behavior pass rate second, operational simplicity
last) applies to eligible candidates only; the transformer lost on eligibility, not
on scores. It remains reproducible with `uv sync --locked --extra transformer`
(research use only; GPU resource gate applies).

## Limitations

- **Synthetic-data ceiling**: all headline metrics come from a controlled synthetic
  corpus; they quantify reproduction quality of a generated task, not real-world
  sentiment accuracy (see the honest-read note above).
- **Behavioral weaknesses** (evidence: 37/60 failed challenge cases, grouped in
  `reports/error_analysis.md`): negation and slang negation, mixed/conflicting
  aspects, restrained sarcasm, direction-of-change pairs, unseen brand names,
  diacritics-free text.
- **Confidence calibration is class-conditional**: sigmoid calibration improved
  proper scores on OOF but the model still emits confidently wrong labels on
  negated/restructured input (e.g. `dir-10` delta 0.6633 in the wrong direction).
- **Monolingual, single domain**: standard written Vietnamese, automotive aspects
  only; no dialect breadth beyond the committed slang rules.
- **Reproducibility caveat (deferred M1 from Task 12)**: the service Dockerfile pins
  **mutable tags** for its base images (`python:3.12-slim`, `ghcr.io/astral-sh/uv:0.11.16`).
  Python dependency reproducibility is guaranteed by the committed `uv.lock`, but the
  base OS layer and Python patch level can drift when Docker Hub retags those
  references. A drift-check of the resolved base digests is recommended before any
  rebuild-for-release.

## Ethical risks

- **Overstated quality**: publishing the OOF/frozen-test numbers without the
  synthetic-data disclaimer could mislead buyers of Vietnamese sentiment services;
  this card forbids that use (see honest-read note).
- **Sarcasm/negation blind spots**: systematically misreading negated complaints as
  neutral/positive could silence genuine safety-related owner feedback (brake,
  airbag, recall phrasing appears in the challenge failures).
- **No personal data, but real-name brands**: the corpus mentions real manufacturers;
  model outputs must not be presented as those brands' official quality assessments.
- **License containment**: the transformer checkpoint may not leave
  research/educational use; serving it commercially would breach the pinned model
  card terms.
