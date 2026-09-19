# Transformer champion export

## Protocol

- Fit on **all 720 development IDs** (the same champion protocol as the baseline: one fit, no epoch selection).
- TrainingArguments mirror the persisted cross-validation folds: learning_rate=2e-5, per_device_train_batch_size=16, num_train_epochs=4, weight_decay=0.01, warmup_ratio=0.10, fp16 when CUDA is available.
- `eval_strategy="no"`, `save_strategy="no"`: with selection complete there is no legitimate validation split and no epoch checkpoint to pick.
- The frozen test set (180 IDs) was never touched; seed 42 matches the persisted split manifest.
- Serving is the raw softmax of the fine-tuned head; no calibration was fitted.

## Frozen inputs

- Dataset: `data/dataset.csv` (sha256:1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043)
- Split manifest: `data/split_manifest.json` (SHA-256 `sha256:026716cd132dd799a1eae17365fa19151b222333713394f67f1fdc6cd9265a4e`)
- Source provenance: `runs/transformer/source.json` (repository `Qualcomm-AI-Research/BamiBERT`, revision `57bc1340debbe4e348ec549047a763caebe4a977`, model-card SHA-256 `6b23dbeffa7ed1e3b4ccf95fb5d6b9eb2f8e8041b334042f54f9c3926c128b61`)
- Installed transformers: `5.5.0`

## Exported artifact

- Directory: `artifacts/transformer/champion` (6 payload files, gitignored — reproduce with the command below)
- Payload checksum: `sha256:3e08c1823af18e8fcc68fcd3c16c54828d5385733026f8071decd2b69641896e`
- Max input tokens: 2048
- Git revision at export: `3ebcc6e93eb39efe2ddf16abe51b768168a41b15`
- License: `bsd-3-clause-clear, other (qualcomm-responsible-ai-license); research/educational intended use; coursework submission, not production deployment`

### Intended use from the model card (verbatim)

> This model is released under the BSD 3-Clause Clear license and the Qualcomm responsible AI license: https://www.qualcomm.com/site/responsible-ai-license

## Serving

```bash
MAT_MODEL_BACKEND=transformer uv run --locked python main.py
```

A fixed `transformer` backend never falls back to the baseline: a missing or corrupt artifact fails startup loudly. The default `baseline` backend keeps serving the linear champion exactly as before.

## Reproduction

```bash
HF_HUB_OFFLINE=1 uv run --locked python -m scripts.release_select_champion export-transformer --model-id Qualcomm-AI-Research/BamiBERT --data data/dataset.csv --splits data/split_manifest.json --source runs/transformer/source.json --artifact-dir artifacts/transformer/champion --work-dir artifacts/transformer/.champion-work --report reports/transformer-champion.md --seed 42
```

The champion reuses `runs/transformer/source.json` (revision-pinned, already resolved once) so the re-run needs no Hub access.

