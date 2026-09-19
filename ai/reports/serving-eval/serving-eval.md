# Đánh giá serving hai backend trên hai tập test

Ngày đo: 2026-09-20 · Dataset chốt: `1675769a…` · API: `POST /predict` qua uvicorn
bản địa (cùng code path với container). Bằng chứng request/response nguyên vẹn
nằm cùng thư mục này, 4 file JSONL — mỗi dòng: `{id, expected, request, response}`.

## Thiết lập

| Thành phần | Giá trị |
| --- | --- |
| Backend A | `baseline` (TF-IDF + LogisticRegression, sigmoid calibration, payload `997d2cc2…`) |
| Backend B | `transformer` (BamiBERT fine-tune, `MAT_MODEL_BACKEND=transformer`, BamiBERT revision `57bc1340…`) |
| Tập 1 — frozen-test | 180 dòng `test_ids` từ `data/split_manifest.json`, chưa từng tham gia train/chọn model |
| Tập 2 — challenge | 74 dòng `data/challenge_set.csv` (MFT/DIR/INV — phủ định, cảm xúc trộn, bất biến) |
| Giao thức | Mỗi câu gửi 1 request; không retry, không chỉnh gì response |

## Kết quả — macro-F1 (nhãn vàng có sẵn trong từng tập)

| Backend | frozen-test (180) | challenge (74) |
| --- | ---: | ---: |
| baseline | 0.6330 | 0.5144 |
| **transformer** | **0.7727** | **0.8065** |

Per-label F1:

| Tập | Backend | negative | neutral | positive |
| --- | --- | ---: | ---: | ---: |
| frozen-test | baseline | 0.661 | 0.609 | 0.629 |
| frozen-test | transformer | 0.814 | 0.729 | 0.775 |
| challenge | baseline | 0.566 | 0.320 | 0.657 |
| challenge | transformer | 0.905 | 0.643 | 0.872 |

Đổi đúng/đổi sai giữa hai backend (so trên cùng tập):

| Tập | transformer ✓ / baseline ✗ | baseline ✓ / transformer ✗ |
| --- | ---: | ---: |
| frozen-test | 44 | 19 |
| challenge | 24 | 4 |

## Kiểm tính nhất quán của phép đo

- Baseline trên frozen-test qua API: **0.6330** — khớp đúng con số frozen-test
  chính thức trong `reports/model-selection.md` (0.63296). Harness đo trung thực,
  không có chênh lệch giữa đường API và đường batch.

## Phát hiện chính — và vì sao vẫn phải đọc cùng CV

Artifact transformer đang serve đo **cao hơn baseline rõ rệt trên cả hai tập**,
kể cả tập frozen chưa từng thấy. Nhưng evidence chọn model (5-fold CV,
`reports/model-selection.md`) lại xếp transformer **thua** (OOF gộp 0.5808 so với
0.6335 calibrated). Hai điều này cùng đúng, giải thích:

- **Phương sai rất lớn**: F1 từng fold của BamiBERT là 0.34 / 0.44 / 0.62 / 0.65 /
  0.75 — với 576 dòng train mỗi fold, vài seed "xấu" kéo mean xuống. CV đo
  *kỳ vọng* của quy trình; export đo *một lượt rút cụ thể*.
- Export train trên đủ 720 dòng dev (CV chỉ 576/fold) — thêm 25% dữ liệu có ý
  nghĩa ở vùng dữ liệu nhỏ này.
- Baseline thì ngược lại: cực kỳ ổn định giữa các fold — nên gates vẫn chọn nó
  làm champion theo tiêu chí độ tin cậy.

**Hệ quả trung thực cần ghi rõ**: con số 0.77/0.80 là đặc tính của *artifact
đang được pin* (revision `894a3de0…` trên Hub). Nếu export lại bằng seed khác,
kết quả có thể xấu đi đáng kể theo phân phối CV. Không tuyên bố transformer
"tốt hơn" ngoài phạm vi artifact này.

## Điểm mạnh/điểm yếu nổi bật của transformer (từ ví dụ thật)

Xử tốt thứ baseline bỏ sót:
- Phủ định: *"Xe không êm, chạy qua gờ giảm tốc là dội vào ghế."* → negative ✓
  (baseline: positive ✗)
- Cảm xúc trộn: *"Nội thất đẹp nhưng phanh trễ khiến mình không yên tâm."* →
  negative ✓ (baseline: positive ✗)
- Đánh giá có số liệu: *"Xăng của BMW 520i khoảng 7.5 lít/100km…"* → positive ✓

Còn yếu ở neutral và emoji:
- *"Ghế sau chật khi đủ ba người."* → neutral ✓ kỳ vọng, transformer: neutral
  là nhãn yếu nhất cả hai tập (F1 0.643 trên challenge).
- *"Chạy đường dài vẫn thoải mái 🙂"* → positive, transformer: neutral ✗
  (emoji ngoài phân phối — đúng như expected của challenge inv-11).

## Hạn chế

1. Dữ liệu synthetic đã duyệt tay — không phản ánh hiệu suất ngoài đời thật.
2. License Qualcomm responsible AI: **research/educational only** — đúng phạm vi
   bài tập, không dùng làm sản phẩm thương mại (xem `reports/transformer-license-audit.md`).
3. Challenge 74 câu là bộ nhỏ, đo được hành vi định tính chứ không phải benchmark thống kê.

## Tái lập

```bash
cd ai
# 1. Chạy server từng backend (đổi MAT_MODEL_BACKEND)
MAT_MODEL_BACKEND=transformer MAT_TRANSFORMER_ARTIFACT_DIR=artifacts/transformer/champion \
  uv run --frozen --no-sync uvicorn main:app --host 127.0.0.1 --port 8123 &
# 2. Gửi 2 tập qua API (harness ở ngoài repo, ghi JSONL req/res)
uv run --locked python /home/faris/repro/mat-dataset-v2/serving_eval.py \
  --port 8123 --backend transformer --out-dir reports/serving-eval
# 3. Lặp cho baseline — metrics trong chính báo cáo này tính từ 4 file JSONL bằng
#    sklearn.metrics (macro-F1, labels = negative/neutral/positive).
```
