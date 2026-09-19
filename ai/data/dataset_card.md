# Dataset card — corpus controlled-v1 (tiếng Việt, ô tô)

## Nó là gì

`data/dataset.csv`: **900 câu synthetic tiếng Việt** về cảm xúc chủ xe, đã rà nhãn
toàn bộ. Không phải bình luận thật, không có dữ liệu cá nhân.

| Con số | Giá trị |
| --- | ---: |
| Tổng dòng | 900 |
| Mỗi nhãn (negative / neutral / positive) | 300 / 300 / 300 |
| Nhóm ý kiến gốc (`canonical_group_id`) | 150 nhóm × đúng 6 biến thể |
| Độ khó (easy / medium / hard) | 300 / 300 / 300 |
| Khía cạnh (engine, fuel, design, comfort, after_sales) | 180 dòng mỗi cái |
| Dòng đã duyệt (`review_status=approved`) | 900 / 900 |

Cách sinh (lịch sử v1): bộ sinh key-based đã nghỉ hưu —
3 nhãn × 10 hãng × 5 khía cạnh = 150 ý kiến gốc, mỗi ý kiến nở thành 6 kiểu
diễn đạt (`review`, `comment`, `comparison`, `question`, `short`, `conversational`).
Vì chỉ có 15 câu lõi × 6 khung câu nên văn bản **lặp mẫu nhiều** —
đây là giới hạn đã biết, không phải lỗi số liệu.

## Nhiễu (330/900 dòng, 36,67%)

| Loại nhiễu | Dòng |
| --- | ---: |
| missing_diacritics | 104 |
| code_switching / emoji / repeated_character | 55 mỗi loại |
| brand_alias | 48 |
| teencode | 13 |

Nhiễu chỉ đổi mặt chữ, không đổi nhãn. `raw_text` là gốc bất biến;
`normalized_text` là view dẫn xuất để kiểm chứng.

## Review và chia tập

* Rà thủ công **6 đợt × 150 dòng** (khớp nhãn, hãng/khía cạnh, giữ ý định khi
  có nhiễu, không nội dung cá nhân/độc hại). Kết quả: duyệt 900/900.
* `data/split_manifest.json` (seed 42) chia theo nhóm ý kiến gốc:
  **720 dòng dev / 5 fold** (mỗi fold 576 train + 144 val),
  **180 dòng frozen test** / 30 nhóm. Dev ∩ frozen = 0 nhóm chung.
* `data/challenge_set.csv`: 60 ca hành vi riêng (phủ định, châm biếm,
  khen–chê lẫn), không trộn vào điểm aggregate.

## Checksum (giá trị frozen, kiểm tra lại được)

| File | SHA-256 |
| --- | --- |
| `data/dataset.csv` | `1675769ac6872f2fff06ab2b4f6ae9f2458a6d7b21f3c59ea38472d2d5c30043` |
| `data/split_manifest.json` | `026716cd132dd799a1eae17365fa19151b222333713394f67f1fdc6cd9265a4e` |
| `data/challenge_set.csv` | `2416e408946d5f66e9f3b96a0c02f635227ac5d78422447ef8ad967a6f300024` |

Kiểm tra: `cd ai && sha256sum data/dataset.csv data/split_manifest.json data/challenge_set.csv`.

## Dùng / không dùng cho việc gì

* Dùng: train, hiệu chỉnh, đánh giá model trong repo này; soi hiện tượng ngôn ngữ
  qua challenge set; tái lập mọi con số từ artifact đã commit.
* Không dùng: benchmark sentiment thật ngoài đời; ước lượng traffic/dân số;
  ý kiến chủ xe thật (không có người thật nào ở đây).
* Không sửa trực tiếp file frozen — dữ liệu mới phải có batch và manifest riêng.

## Giới hạn đã biết

Tiếng Việt chuẩn, thiên chính tả miền Bắc; teencode/dialect ngoài vài quy tắc
(`ko`, `khum`, `nha`…) không được phủ. Bỏ dấu chỉ ở mức transform máy;
code-switching chỉ là từ tiếng Anh ngắn chen vào. Câu phức tạp ngoài phân phối
(phủ định, sarcasm) xem thêm `reports/error_analysis.md`.
