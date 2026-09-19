# Baseline evaluation

## Evaluation scope

- Selected prediction view: `raw`
- Challenge inference: baseline reconstructed from the selected text view
- OOF rows: 720
- Unique OOF IDs: 720
- Behavioral challenge cases are excluded from aggregate metrics and model fitting.

## Fold distribution

| Fold | Count | Macro-F1 |
| --- | ---: | ---: |
| 0 | 144 | 0.644573 |
| 1 | 144 | 0.667543 |
| 2 | 144 | 0.638805 |
| 3 | 144 | 0.598642 |
| 4 | 144 | 0.638015 |

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.637804 |
| Log loss | 0.828758 |
| Multiclass Brier | 0.481512 |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.676991 | 0.637500 | 0.656652 | 240 |
| neutral | 0.609562 | 0.637500 | 0.623218 | 240 |
| positive | 0.629630 | 0.637500 | 0.633540 | 240 |

## Confusion matrix

Rows are actual labels; columns are predictions in fixed order.

| Actual \ Predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 153 | 45 | 42 |
| neutral | 39 | 153 | 48 |
| positive | 34 | 53 | 153 |

## Slice metrics

### noise_types

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| brand_alias | 98 | 0.600166 | 0.883877 | 0.512896 |
| code_switching | 38 | 0.909325 | 0.522228 | 0.276939 |
| elongation | 22 | 0.570806 | 0.484961 | 0.252117 |
| emoji | 68 | 0.756786 | 0.630789 | 0.350476 |
| missing_diacritics | 25 | 0.335979 | 0.908102 | 0.548525 |
| teencode | 70 | 0.646547 | 0.731555 | 0.428330 |

### aspect

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| an toàn | 84 | 0.543203 | 0.891785 | 0.532951 |
| công nghệ & tiện nghi | 96 | 0.686994 | 0.828125 | 0.477827 |
| dịch vụ hậu mãi | 88 | 0.686772 | 0.797071 | 0.460868 |
| giá cả | 88 | 0.510262 | 0.981494 | 0.565945 |
| mức tiêu hao nhiên liệu | 93 | 0.649237 | 0.785506 | 0.460311 |
| nội thất | 92 | 0.660213 | 0.795303 | 0.458951 |
| thiết kế | 91 | 0.625493 | 0.788244 | 0.460582 |
| động cơ | 88 | 0.692084 | 0.770821 | 0.440277 |

### style

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| comment | 104 | 0.639472 | 0.759368 | 0.439712 |
| comparison | 98 | 0.535418 | 0.996462 | 0.584995 |
| conversational | 111 | 0.590160 | 0.888629 | 0.521400 |
| question | 106 | 0.426667 | 0.901445 | 0.533128 |
| review | 175 | 0.702047 | 0.741472 | 0.423753 |
| short | 126 | 0.721191 | 0.762935 | 0.437183 |

### difficulty

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| easy | 338 | 0.713098 | 0.755318 | 0.433208 |
| hard | 48 | 0.467624 | 0.966592 | 0.582642 |
| medium | 334 | 0.575651 | 0.883270 | 0.515861 |

## Behavioral challenge suite

Overall: 37/74 (0.500000).

| Kind | Passed | Count | Pass rate |
| --- | ---: | ---: | ---: |
| DIR | 9 | 22 | 0.409091 |
| INV | 11 | 23 | 0.478261 |
| MFT | 17 | 29 | 0.586207 |

### Phenomenon results

| Phenomenon | Passed | Count | Pass rate |
| --- | ---: | ---: | ---: |
| aspect_conflict | 4 | 7 | 0.571429 |
| brand_abbreviation | 2 | 3 | 0.666667 |
| casing | 2 | 3 | 0.666667 |
| comparison | 2 | 6 | 0.333333 |
| emoji | 5 | 5 | 1.000000 |
| missing_diacritics | 1 | 6 | 0.166667 |
| mixed_sentiment | 4 | 5 | 0.800000 |
| negation | 3 | 9 | 0.333333 |
| punctuation | 2 | 3 | 0.666667 |
| restrained_sarcasm | 4 | 7 | 0.571429 |
| slang_negation | 1 | 5 | 0.200000 |
| spacing | 2 | 2 | 1.000000 |
| specification | 2 | 4 | 0.500000 |
| unicode_normalization | 0 | 2 | 0.000000 |
| unseen_model_names | 3 | 7 | 0.428571 |

## Challenge failures

| ID | Kind | Phenomenon | Detail |
| --- | --- | --- | --- |
| mft-01 | MFT | negation | predicted=positive; expected=negative |
| mft-02 | MFT | negation | predicted=neutral; expected=positive |
| mft-06 | MFT | aspect_conflict | predicted=negative; expected=positive |
| mft-07 | MFT | comparison | predicted=negative; expected=positive |
| mft-08 | MFT | comparison | predicted=neutral; expected=negative |
| mft-10 | MFT | restrained_sarcasm | predicted=neutral; expected=negative |
| mft-12 | MFT | unseen_model_names | predicted=negative; expected=positive |
| mft-17 | MFT | slang_negation | predicted=positive; expected=negative |
| mft-19 | MFT | unseen_model_names | predicted=negative; expected=neutral |
| mft-20 | MFT | comparison | predicted=negative; expected=positive |
| inv-03 | INV | slang_negation | negative->negative |
| inv-05 | INV | casing | negative->negative |
| inv-09 | INV | punctuation | negative->negative |
| inv-13 | INV | missing_diacritics | positive->negative |
| inv-14 | INV | missing_diacritics | positive->neutral |
| inv-15 | INV | unicode_normalization | positive->negative |
| inv-16 | INV | unicode_normalization | negative->neutral |
| inv-18 | INV | brand_abbreviation | negative->neutral |
| inv-19 | INV | missing_diacritics | positive->negative |
| inv-20 | INV | slang_negation | positive->positive |
| dir-01 | DIR | negation | delta=0.1820 |
| dir-02 | DIR | negation | delta=0.0802 |
| dir-03 | DIR | mixed_sentiment | delta=0.3651 |
| dir-06 | DIR | restrained_sarcasm | delta=0.1207 |
| dir-08 | DIR | slang_negation | delta=-0.0358 |
| dir-10 | DIR | comparison | delta=0.1273 |
| dir-12 | DIR | unseen_model_names | delta=-0.0827 |
| dir-13 | DIR | aspect_conflict | delta=0.1770 |
| dir-14 | DIR | aspect_conflict | delta=0.0323 |
| dir-17 | DIR | negation | delta=0.0001 |
| dir-19 | DIR | negation | delta=0.1145 |
| dir-20 | DIR | unseen_model_names | delta=-0.0999 |
| mft-22 | MFT | specification | predicted=negative; expected=neutral |
| mft-24 | MFT | specification | predicted=negative; expected=neutral |
| inv-21 | INV | missing_diacritics | positive->negative |
| inv-23 | INV | missing_diacritics | negative->negative |
| dir-21 | DIR | restrained_sarcasm | delta=-0.2496 |

## Source checksums

| Artifact | SHA-256 |
| --- | --- |
| predictions | `7d7878bdd792c40a0b1dbd5348439ab258959bc48339a2b50eb1cfc0858ce784` |
| dataset | `1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043` |
| challenge_set | `2416e408946d5f66e9f3b96a0c02f635227ac5d78422447ef8ad967a6f300024` |
