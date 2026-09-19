# Transformer evaluation

## Evaluation scope

- Selected prediction view: `single`
- Challenge inference: mean probabilities persisted from the five revision-pinned transformer fold models
- OOF rows: 720
- Unique OOF IDs: 720
- Behavioral challenge cases are excluded from aggregate metrics and model fitting.

## Fold distribution

| Fold | Count | Macro-F1 |
| --- | ---: | ---: |
| 0 | 144 | 0.444746 |
| 1 | 144 | 0.337915 |
| 2 | 144 | 0.749231 |
| 3 | 144 | 0.645876 |
| 4 | 144 | 0.624797 |

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.580778 |
| Log loss | 0.919399 |
| Multiclass Brier | 0.544352 |

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| negative | 0.725146 | 0.516667 | 0.603406 | 240 |
| neutral | 0.595506 | 0.441667 | 0.507177 | 240 |
| positive | 0.520216 | 0.804167 | 0.631751 | 240 |

## Confusion matrix

Rows are actual labels; columns are predictions in fixed order.

| Actual \ Predicted | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 124 | 42 | 74 |
| neutral | 30 | 106 | 104 |
| positive | 17 | 30 | 193 |

## Slice metrics

### noise_types

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| brand_alias | 98 | 0.595075 | 0.895095 | 0.530960 |
| code_switching | 38 | 0.531624 | 0.901334 | 0.527088 |
| elongation | 22 | 0.430556 | 0.975559 | 0.579834 |
| emoji | 68 | 0.631898 | 0.822438 | 0.473910 |
| missing_diacritics | 25 | 0.251341 | 0.992580 | 0.605399 |
| teencode | 70 | 0.599455 | 0.903558 | 0.536834 |

### aspect

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| an toàn | 84 | 0.546606 | 0.933911 | 0.552626 |
| công nghệ & tiện nghi | 96 | 0.599058 | 0.893388 | 0.525474 |
| dịch vụ hậu mãi | 88 | 0.551464 | 0.896063 | 0.529127 |
| giá cả | 88 | 0.545139 | 0.963311 | 0.574805 |
| mức tiêu hao nhiên liệu | 93 | 0.608154 | 0.939729 | 0.552831 |
| nội thất | 92 | 0.585942 | 0.897389 | 0.528007 |
| thiết kế | 91 | 0.522944 | 0.949740 | 0.571423 |
| động cơ | 88 | 0.665879 | 0.883500 | 0.521951 |

### style

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| comment | 104 | 0.612103 | 0.886892 | 0.520898 |
| comparison | 98 | 0.525427 | 0.978351 | 0.574568 |
| conversational | 111 | 0.520952 | 0.932411 | 0.557655 |
| question | 106 | 0.501755 | 0.905554 | 0.538204 |
| review | 175 | 0.515983 | 0.902094 | 0.530541 |
| short | 126 | 0.542754 | 0.924599 | 0.552842 |

### difficulty

| Slice | Count | Macro-F1 | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| easy | 338 | 0.586660 | 0.900119 | 0.532870 |
| hard | 48 | 0.457419 | 0.933250 | 0.545496 |
| medium | 334 | 0.566288 | 0.936920 | 0.555806 |

## Behavioral challenge suite

Overall: 34/74 (0.459459).

| Kind | Passed | Count | Pass rate |
| --- | ---: | ---: | ---: |
| DIR | 6 | 22 | 0.272727 |
| INV | 8 | 23 | 0.347826 |
| MFT | 20 | 29 | 0.689655 |

### Phenomenon results

| Phenomenon | Passed | Count | Pass rate |
| --- | ---: | ---: | ---: |
| aspect_conflict | 3 | 7 | 0.428571 |
| brand_abbreviation | 1 | 3 | 0.333333 |
| casing | 2 | 3 | 0.666667 |
| comparison | 1 | 6 | 0.166667 |
| emoji | 4 | 5 | 0.800000 |
| missing_diacritics | 3 | 6 | 0.500000 |
| mixed_sentiment | 2 | 5 | 0.400000 |
| negation | 2 | 9 | 0.222222 |
| punctuation | 2 | 3 | 0.666667 |
| restrained_sarcasm | 5 | 7 | 0.714286 |
| slang_negation | 1 | 5 | 0.200000 |
| spacing | 0 | 2 | 0.000000 |
| specification | 3 | 4 | 0.750000 |
| unicode_normalization | 0 | 2 | 0.000000 |
| unseen_model_names | 5 | 7 | 0.714286 |

## Challenge failures

| ID | Kind | Phenomenon | Detail |
| --- | --- | --- | --- |
| mft-02 | MFT | negation | predicted=neutral; expected=positive |
| mft-04 | MFT | mixed_sentiment | predicted=negative; expected=positive |
| mft-06 | MFT | aspect_conflict | predicted=neutral; expected=positive |
| mft-07 | MFT | comparison | predicted=neutral; expected=positive |
| mft-08 | MFT | comparison | predicted=neutral; expected=negative |
| mft-17 | MFT | slang_negation | predicted=positive; expected=negative |
| inv-01 | INV | brand_abbreviation | positive->neutral |
| inv-03 | INV | slang_negation | neutral->neutral |
| inv-04 | INV | slang_negation | negative->neutral |
| inv-05 | INV | casing | neutral->positive |
| inv-07 | INV | spacing | neutral->neutral |
| inv-08 | INV | spacing | neutral->neutral |
| inv-09 | INV | punctuation | positive->neutral |
| inv-11 | INV | emoji | neutral->neutral |
| inv-13 | INV | missing_diacritics | neutral->neutral |
| inv-14 | INV | missing_diacritics | negative->neutral |
| inv-15 | INV | unicode_normalization | positive->negative |
| inv-16 | INV | unicode_normalization | negative->neutral |
| inv-17 | INV | unseen_model_names | neutral->neutral |
| inv-18 | INV | brand_abbreviation | neutral->neutral |
| inv-19 | INV | missing_diacritics | neutral->negative |
| dir-01 | DIR | negation | delta=0.1262 |
| dir-02 | DIR | negation | delta=0.0977 |
| dir-05 | DIR | comparison | delta=0.1401 |
| dir-06 | DIR | restrained_sarcasm | delta=0.1652 |
| dir-08 | DIR | slang_negation | delta=0.0302 |
| dir-09 | DIR | aspect_conflict | delta=0.1869 |
| dir-10 | DIR | comparison | delta=0.1111 |
| dir-11 | DIR | mixed_sentiment | delta=0.1073 |
| dir-12 | DIR | unseen_model_names | delta=0.1422 |
| dir-14 | DIR | aspect_conflict | delta=0.1142 |
| dir-15 | DIR | comparison | delta=0.0736 |
| dir-16 | DIR | mixed_sentiment | delta=0.1478 |
| dir-17 | DIR | negation | delta=0.0716 |
| dir-18 | DIR | aspect_conflict | delta=0.0660 |
| dir-19 | DIR | negation | delta=-0.0047 |
| mft-21 | MFT | specification | predicted=positive; expected=neutral |
| dir-21 | DIR | restrained_sarcasm | delta=-0.0354 |
| mft-28 | MFT | negation | predicted=neutral; expected=positive |
| mft-29 | MFT | negation | predicted=neutral; expected=positive |

## Source checksums

| Artifact | SHA-256 |
| --- | --- |
| predictions | `3ba362f3470951c19bf1d757a55d2d2bfe122abcffaf9b777520cb1bbdfaafd8` |
| dataset | `1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043` |
| challenge_set | `2416e408946d5f66e9f3b96a0c02f635227ac5d78422447ef8ad967a6f300024` |
| transformer_challenge_predictions | `5dab35a52a4b6b9462352962f27aca6a63b2cf10fe1dd3f092e6dfb9692eaed3` |
