# Vietnamese robustness suite — observational agreement report

- suite: vietnamese-robustness
- suite version: 1
- cases: 90 (15 categories x 6)
- model: backend='linear' version='1.0.0' (baseline serving)
- snapshot hash (canonical, request ids excluded): d5caa2cdcac70cc929f066854017eae135920ed88841d49f311429377ffa0f3c

## Overall agreement (observational)

- human/model label agreement: 53/90 (58.9%)

## Agreement by category

| category | cases | agree | disagree | disagreement rate |
|---|---:|---:|---:|---:|
| regional-north | 6 | 4 | 2 | 33.3% |
| regional-central | 6 | 3 | 3 | 50.0% |
| regional-south | 6 | 3 | 3 | 50.0% |
| channel-automotive-forum | 6 | 4 | 2 | 33.3% |
| channel-social-video-comment | 6 | 5 | 1 | 16.7% |
| channel-ecommerce-review | 6 | 3 | 3 | 50.0% |
| channel-sales-service-chat | 6 | 4 | 2 | 33.3% |
| teencode-abbreviation | 6 | 4 | 2 | 33.3% |
| missing-diacritics-typos | 6 | 3 | 3 | 50.0% |
| code-switching | 6 | 4 | 2 | 33.3% |
| negation-scope | 6 | 2 | 4 | 66.7% |
| mixed-aspects | 6 | 3 | 3 | 50.0% |
| sarcasm-idiom | 6 | 3 | 3 | 50.0% |
| automotive-jargon | 6 | 4 | 2 | 33.3% |
| target-ambiguity | 6 | 4 | 2 | 33.3% |

## Disagreement details (case ID only; no reviewer personal data)

- AF-02 (channel-automotive-forum): human=positive model=negative
- AF-04 (channel-automotive-forum): human=neutral model=negative
- AJ-01 (automotive-jargon): human=positive model=negative
- AJ-03 (automotive-jargon): human=neutral model=positive
- CS-01 (code-switching): human=positive model=negative
- CS-03 (code-switching): human=neutral model=negative
- DT-01 (missing-diacritics-typos): human=positive model=negative
- DT-02 (missing-diacritics-typos): human=positive model=negative
- DT-05 (missing-diacritics-typos): human=negative model=neutral
- EC-01 (channel-ecommerce-review): human=positive model=negative
- EC-02 (channel-ecommerce-review): human=positive model=negative
- EC-03 (channel-ecommerce-review): human=neutral model=negative
- MA-01 (mixed-aspects): human=positive model=negative
- MA-03 (mixed-aspects): human=neutral model=negative
- MA-04 (mixed-aspects): human=neutral model=positive
- NS-01 (negation-scope): human=positive model=negative
- NS-02 (negation-scope): human=positive model=neutral
- NS-03 (negation-scope): human=neutral model=negative
- NS-06 (negation-scope): human=negative model=positive
- RC-03 (regional-central): human=neutral model=negative
- RC-04 (regional-central): human=neutral model=positive
- RC-06 (regional-central): human=negative model=positive
- RN-03 (regional-north): human=neutral model=negative
- RN-04 (regional-north): human=neutral model=positive
- RS-02 (regional-south): human=positive model=negative
- RS-03 (regional-south): human=neutral model=positive
- RS-04 (regional-south): human=neutral model=positive
- SC-02 (channel-sales-service-chat): human=positive model=negative
- SC-04 (channel-sales-service-chat): human=neutral model=positive
- SI-02 (sarcasm-idiom): human=positive model=neutral
- SI-04 (sarcasm-idiom): human=neutral model=positive
- SI-06 (sarcasm-idiom): human=negative model=neutral
- SV-01 (channel-social-video-comment): human=positive model=negative
- TA-03 (target-ambiguity): human=neutral model=negative
- TA-04 (target-ambiguity): human=neutral model=negative
- TT-03 (teencode-abbreviation): human=neutral model=negative
- TT-05 (teencode-abbreviation): human=negative model=positive

## Scope

Semantic agreement is observational for the frozen baseline; this report is not
an accuracy claim and never gates a release. Hard gates (schema, corpus overlap,
API contract, snapshot equality) are enforced by the runner exit code.
