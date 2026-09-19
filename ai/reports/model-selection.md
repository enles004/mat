# Model selection

## Scope and frozen inputs

- Recorded: 2026-09-19T20:45:07Z UTC
- Git revision: `3ebcc6e93eb39efe2ddf16abe51b768168a41b15`
- Champion gates below were written before any frozen-test label access.

| Frozen input | SHA-256 |
| --- | --- |
| `data/dataset.csv` | `1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043` |
| `data/split_manifest.json` | `026716cd132dd799a1eae17365fa19151b222333713394f67f1fdc6cd9265a4e` |
| `data/challenge_set.csv` | `2416e408946d5f66e9f3b96a0c02f635227ac5d78422447ef8ad967a6f300024` |

- `configs/models.yaml` selection.calibration_max_macro_f1_drop confirmed as 0.01.

## Baseline calibration gate (cross-fitted over persisted folds)

Per the binding ruling, calibrated OOF scores come from leave-one-fold-out
calibration: for each held-out fold the sigmoid calibrator is fitted only on the
other four folds' OOF outputs, and the held-out fold is scored by that calibrator.
The final calibrator for the exported model is fitted on all OOF outputs via the
five persisted fold pairs. No row is ever calibrated on itself.

| Metric | Native OOF | Cross-fitted calibrated OOF |
| --- | ---: | ---: |
| Log loss | 0.828758 | 0.827033 |
| Multiclass Brier | 0.481512 | 0.478018 |
| Macro-F1 | 0.637804 | 0.633488 |

- Decision: **keep sigmoid calibration** — at least one proper score improved (log loss -0.001725, Brier -0.003493) and macro-F1 dropped by at most 0.01 (change -0.004316).

## Candidate gates

| Gate | baseline | transformer |
| --- | --- | --- |
| Reproducible five-fold run | PASS — five persisted folds over 720 OOF rows, view=raw | PASS — five persisted folds over 720 OOF rows, mean fold macro-F1 0.560513 |
| No group leakage | PASS — 0 manifest validation errors; 720 development groups disjoint from 180 frozen-test groups; fold group isolation=True | PASS — 0 manifest validation errors; 720 development groups disjoint from 180 frozen-test groups; fold group isolation=True |
| All required labels | PASS — OOF truth and metrics cover negative/neutral/positive | PASS — OOF truth and metrics cover negative/neutral/positive |
| Finite probabilities | PASS — OOF probabilities finite, in [0, 1], rows sum to one | PASS — OOF probabilities finite, in [0, 1], rows sum to one |
| Behavior report present | PASS — 37/74 challenges passed (baseline-evaluation.json) | PASS — 34/74 challenges passed (transformer-evaluation.json) |
| Artifact export supported | PASS — joblib-serializable scikit-learn pipeline | FAIL — not demonstrated: no transformer payload exported (Task 8 persisted metrics/predictions only) |
| Exact source revision recorded | PASS — git revision 3ebcc6e93eb39efe2ddf16abe51b768168a41b15 | PASS — BamiBERT revision 57bc1340debbe4e348ec549047a763caebe4a977 |
| License/intended use approved | PASS — trained in this repository on reviewed synthetic data | FAIL — model card intended use: 'The model is intended for research and educational purposes.'; deployment requires documented approval of the exact terms (see reports/transformer-license-audit.md) |
| Macro-F1 (development OOF) | native OOF 0.637804 matches 0.637804 from baseline-evaluation.json; cross-fitted calibrated 0.633488; effective 0.633488 | OOF 0.580778 matches 0.580778 from transformer-evaluation.json |
| Behavior pass rate | 0.500000 | 0.459459 |
| License status | approved: internal-use-only (in-repo trained artifact, no external terms) | research-only: Qualcomm responsible AI license restricts to research and educational use |

## Ranking and champion

Ranking is macro-F1 first, behavior pass rate second, baseline operational
simplicity as the final tie-break. Only eligible candidates may win.

| Candidate | Eligible | Macro-F1 | Behavior pass rate | Ranking tuple |
| --- | --- | ---: | ---: | --- |
| baseline | True | 0.633488 | 0.500000 | (0.633488, 0.500000, True) |
| transformer | False | 0.580778 | 0.459459 | (0.580778, 0.459459, False) |

- `transformer` is ineligible: research-only: Qualcomm responsible AI license restricts to research and educational use.

**Champion: baseline** — selected under macro-F1 first, behavior pass
rate second, baseline operational simplicity last, among eligible candidates only.

## Frozen-test evaluation (single run after the gate write)

The final pipeline was fitted on all 720 development IDs with the selected view
and calibration, then evaluated on the 180 frozen test rows exactly once. No
model-selection decision was revisited from these numbers.

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.632962 |
| Log loss | 0.753297 |
| Multiclass Brier | 0.441639 |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.655738 | 0.666667 | 0.661157 | 60 |
| neutral | 0.636364 | 0.583333 | 0.608696 | 60 |
| positive | 0.609375 | 0.650000 | 0.629032 | 60 |

## Confusion matrix

Rows are actual labels; columns are predictions in fixed order.

| Actual \\ Predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 40 | 8 | 12 |
| neutral | 12 | 35 | 13 |
| positive | 9 | 12 | 39 |

## Slice metrics

### noise_types

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| brand_alias | 17 | 0.698834 | 0.684817 | 0.383057 |
| emoji | 17 | 0.686090 | 0.764631 | 0.431111 |
| teencode | 17 | 0.571429 | 0.798315 | 0.481012 |

### aspect

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| an toàn | 28 | 0.708333 | 0.630893 | 0.370934 |
| công nghệ & tiện nghi | 16 | 0.467033 | 0.786794 | 0.481974 |
| dịch vụ hậu mãi | 25 | 0.711111 | 0.687151 | 0.383704 |
| giá cả | 24 | 0.625926 | 0.753562 | 0.429447 |
| mức tiêu hao nhiên liệu | 19 | 0.383957 | 0.914620 | 0.569174 |
| nội thất | 21 | 0.697354 | 0.650070 | 0.363346 |
| thiết kế | 21 | 0.571429 | 0.809426 | 0.472030 |
| động cơ | 26 | 0.643665 | 0.848011 | 0.505414 |

### style

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| comment | 34 | 0.738057 | 0.671748 | 0.380367 |
| comparison | 22 | 0.372222 | 0.848764 | 0.500955 |
| conversational | 21 | 0.579757 | 0.858615 | 0.501360 |
| question | 29 | 0.464469 | 0.781986 | 0.475555 |
| review | 46 | 0.630988 | 0.723856 | 0.433773 |
| short | 28 | 0.609195 | 0.716976 | 0.402441 |

### difficulty

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| easy | 93 | 0.582635 | 0.791031 | 0.466512 |
| hard | 11 | 0.316239 | 0.924006 | 0.569857 |
| medium | 76 | 0.699937 | 0.682415 | 0.392644 |

## Error examples

| ID | Actual | Predicted | negative | neutral | positive |
| --- | --- | --- | ---: | ---: | ---: |
| curated-v2-0005 | positive | negative | 0.5842 | 0.1620 | 0.2538 |
| curated-v2-0006 | positive | neutral | 0.2893 | 0.3924 | 0.3183 |
| curated-v2-0026 | positive | neutral | 0.0521 | 0.6161 | 0.3318 |
| curated-v2-0031 | negative | positive | 0.2689 | 0.2739 | 0.4572 |
| curated-v2-0044 | negative | positive | 0.2080 | 0.1592 | 0.6328 |
| curated-v2-0045 | neutral | positive | 0.1101 | 0.4427 | 0.4472 |
| curated-v2-0081 | positive | neutral | 0.0831 | 0.8501 | 0.0668 |
| curated-v2-0091 | neutral | positive | 0.2485 | 0.2567 | 0.4948 |
| curated-v2-0105 | negative | neutral | 0.3640 | 0.4705 | 0.1655 |
| curated-v2-0109 | positive | neutral | 0.0768 | 0.5353 | 0.3878 |
| curated-v2-0148 | positive | negative | 0.4819 | 0.1661 | 0.3521 |
| curated-v2-0178 | positive | negative | 0.4391 | 0.3563 | 0.2046 |
| curated-v2-0195 | positive | negative | 0.6788 | 0.1550 | 0.1661 |
| curated-v2-0200 | negative | neutral | 0.3423 | 0.5306 | 0.1271 |
| curated-v2-0206 | positive | neutral | 0.0884 | 0.4811 | 0.4305 |
| curated-v2-0211 | negative | neutral | 0.3384 | 0.4470 | 0.2146 |
| curated-v2-0221 | negative | positive | 0.2148 | 0.2174 | 0.5678 |
| curated-v2-0235 | positive | negative | 0.5409 | 0.0895 | 0.3695 |
| curated-v2-0264 | negative | positive | 0.4061 | 0.1096 | 0.4842 |
| curated-v2-0273 | positive | neutral | 0.3313 | 0.4414 | 0.2272 |
| curated-v2-0290 | positive | negative | 0.5381 | 0.2351 | 0.2268 |
| curated-v2-0293 | positive | neutral | 0.2880 | 0.4538 | 0.2582 |
| curated-v2-0315 | neutral | negative | 0.4718 | 0.1864 | 0.3418 |
| curated-v2-0323 | neutral | negative | 0.4488 | 0.4216 | 0.1296 |
| curated-v2-0341 | positive | neutral | 0.3430 | 0.3735 | 0.2835 |
| curated-v2-0344 | neutral | positive | 0.0784 | 0.2825 | 0.6392 |
| curated-v2-0364 | negative | positive | 0.4330 | 0.0882 | 0.4788 |
| curated-v2-0378 | positive | neutral | 0.2345 | 0.5400 | 0.2256 |
| curated-v2-0380 | positive | neutral | 0.0839 | 0.4969 | 0.4192 |
| curated-v2-0382 | neutral | negative | 0.4526 | 0.3751 | 0.1723 |
| curated-v2-0435 | positive | neutral | 0.0768 | 0.5368 | 0.3864 |
| curated-v2-0460 | negative | positive | 0.2241 | 0.3192 | 0.4567 |
| curated-v2-0477 | neutral | negative | 0.6827 | 0.2243 | 0.0931 |
| curated-v2-0484 | negative | positive | 0.1686 | 0.3914 | 0.4400 |
| curated-v2-0496 | negative | neutral | 0.2515 | 0.5115 | 0.2370 |
| curated-v2-0497 | negative | positive | 0.4400 | 0.0898 | 0.4702 |
| curated-v2-0502 | neutral | positive | 0.0355 | 0.4373 | 0.5272 |
| curated-v2-0508 | positive | negative | 0.6091 | 0.2467 | 0.1442 |
| curated-v2-0514 | neutral | negative | 0.6974 | 0.2745 | 0.0280 |
| curated-v2-0516 | neutral | positive | 0.1436 | 0.3959 | 0.4605 |
| curated-v2-0537 | neutral | negative | 0.4606 | 0.3925 | 0.1468 |
| curated-v2-0541 | neutral | positive | 0.0929 | 0.4241 | 0.4830 |
| curated-v2-0544 | neutral | negative | 0.4712 | 0.3283 | 0.2004 |
| curated-v2-0559 | neutral | positive | 0.3067 | 0.3343 | 0.3590 |
| curated-v2-0570 | positive | neutral | 0.0826 | 0.4850 | 0.4324 |
| curated-v2-0585 | negative | positive | 0.1350 | 0.2099 | 0.6551 |
| curated-v2-0589 | negative | positive | 0.4028 | 0.1837 | 0.4135 |
| curated-v2-0596 | positive | negative | 0.4672 | 0.1697 | 0.3631 |
| curated-v2-0619 | neutral | negative | 0.4655 | 0.4282 | 0.1063 |
| curated-v2-0621 | neutral | negative | 0.4497 | 0.2818 | 0.2685 |
| curated-v2-0624 | neutral | negative | 0.4903 | 0.0991 | 0.4106 |
| curated-v2-0637 | neutral | positive | 0.0465 | 0.2328 | 0.7207 |
| curated-v2-0647 | neutral | positive | 0.1883 | 0.1882 | 0.6235 |
| curated-v2-0669 | neutral | negative | 0.6468 | 0.2311 | 0.1221 |
| curated-v2-0691 | neutral | positive | 0.1693 | 0.3130 | 0.5177 |
| curated-v2-0715 | negative | neutral | 0.3083 | 0.3754 | 0.3163 |
| curated-v2-0725 | positive | negative | 0.3995 | 0.2065 | 0.3940 |
| curated-v2-0743 | negative | positive | 0.2300 | 0.3835 | 0.3865 |
| curated-v2-0756 | negative | positive | 0.3530 | 0.1467 | 0.5003 |
| curated-v2-0759 | neutral | positive | 0.0935 | 0.4458 | 0.4607 |
| curated-v2-0764 | neutral | positive | 0.0108 | 0.4327 | 0.5564 |
| curated-v2-0766 | neutral | positive | 0.1515 | 0.3361 | 0.5124 |
| curated-v2-0853 | negative | neutral | 0.2389 | 0.4906 | 0.2704 |
| curated-v2-0855 | negative | neutral | 0.4141 | 0.5521 | 0.0337 |
| curated-v2-0857 | negative | neutral | 0.2429 | 0.3932 | 0.3638 |
| curated-v2-0883 | neutral | negative | 0.5426 | 0.3519 | 0.1055 |

## Latency and environment

| Measurement | Value |
| --- | ---: |
| Per-row predict latency mean | 5.422391 ms |
| Per-row predict latency p50 | 5.409866 ms |
| Per-row predict latency p95 | 5.757632 ms |
| Batch latency (180 rows) | 41.092612 ms |
| Platform | Linux-7.0.0-31-generic-x86_64-with-glibc2.43 |
| CPU | Intel(R) Core(TM) i5-14600KF |
| Python | 3.12.13 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.3 |

## Behavioral re-check on the exported model

37/74 challenges passed (0.500000) when the exported artifact itself was re-run over the reviewed challenge set (development-only evidence, no frozen rows).

| ID | Kind | Detail |
| --- | --- | --- |
| mft-01 | MFT | predicted=positive; expected=negative |
| mft-02 | MFT | predicted=neutral; expected=positive |
| mft-03 | MFT | predicted=neutral; expected=negative |
| mft-06 | MFT | predicted=negative; expected=positive |
| mft-07 | MFT | predicted=negative; expected=positive |
| mft-08 | MFT | predicted=neutral; expected=negative |
| mft-10 | MFT | predicted=neutral; expected=negative |
| mft-12 | MFT | predicted=negative; expected=positive |
| mft-17 | MFT | predicted=positive; expected=negative |
| mft-19 | MFT | predicted=negative; expected=neutral |
| mft-20 | MFT | predicted=negative; expected=positive |
| inv-03 | INV | negative->negative |
| inv-05 | INV | negative->negative |
| inv-09 | INV | negative->negative |
| inv-13 | INV | positive->negative |
| inv-14 | INV | positive->neutral |
| inv-15 | INV | positive->negative |
| inv-16 | INV | negative->neutral |
| inv-18 | INV | neutral->neutral |
| inv-19 | INV | positive->negative |
| inv-20 | INV | neutral->positive |
| dir-01 | DIR | delta=0.1890 |
| dir-02 | DIR | delta=0.0662 |
| dir-03 | DIR | delta=0.3051 |
| dir-06 | DIR | delta=0.0875 |
| dir-08 | DIR | delta=-0.0242 |
| dir-10 | DIR | delta=0.1505 |
| dir-12 | DIR | delta=-0.0866 |
| dir-14 | DIR | delta=0.0444 |
| dir-17 | DIR | delta=-0.0012 |
| dir-19 | DIR | delta=0.0831 |
| dir-20 | DIR | delta=-0.0860 |
| mft-22 | MFT | predicted=negative; expected=neutral |
| mft-24 | MFT | predicted=negative; expected=neutral |
| inv-21 | INV | positive->negative |
| inv-23 | INV | negative->negative |
| dir-21 | DIR | delta=-0.2870 |
## Exported artifact

- Directory: `artifacts/baseline` (payload `model.joblib` plus `manifest.json`)
- Payload checksum: `sha256:997d2cc2545b9fed99f49b4b0f395fe69aa2b0be6dae045685d96b00b7295337`
- Model: `tfidf-word-char-logreg` version 1.0.0 (backend `linear`)
- Labels: negative, neutral, positive
- Preprocessor: tfidf-word-bigram-char_wb-trigram (raw view) (config checksum `sha256:756735fe3ef5f0eac1a6ef38ea32e1d535dcd501c79e9ceef9e491346703bdba`)
- Calibration: method `sigmoid`, fitted on "720 development OOF rows; five persisted fold calibrators ensembled"
- Dataset checksum: `sha256:1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043`
- Git revision: `3ebcc6e93eb39efe2ddf16abe51b768168a41b15`
- License: `internal-use-only`
- Evaluation report: `reports/model-selection.md`
- Max input tokens: not applicable (the linear backend applies no tokenizer limit)

Verify with: `cd ai && uv run --locked python scripts/release_verify_artifact.py verify artifacts/baseline`.
