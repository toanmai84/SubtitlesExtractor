# Khung tự hiệu chuẩn đệ quy (Recursive Self-Calibration)

Tự động dò bộ tham số tốt nhất cho hai chức năng cốt lõi của ứng dụng:

1. **Build phụ đề từ OCR thô** — tối ưu `SubtitleBuilderConfig` sao cho phụ đề
   dựng ra giống phụ đề chuẩn (ground-truth SRT) nhất.
2. **Phát hiện dải ROI tự động** — tối ưu tham số `BBoxAnalyzer` (chạy offline từ
   `*_seraw.json`, không cần video/GPU).

## Triết lý

- **Đệ quy (recursive)**: optimizer quét coordinate-descent trên lưới thô, tìm
  điểm tốt nhất, rồi *thu hẹp biên lưới quanh điểm đó và tự gọi lại* ở độ phân
  giải mịn hơn — dừng khi hết độ sâu hoặc không còn cải thiện.
- **Tự cải tiến (self-improving)**: `RecursiveCalibrator` lưu bộ tham số tốt nhất
  ra JSON. Phiên sau *warm-start* từ trạng thái cũ nên kết quả chỉ tốt lên hoặc
  giữ nguyên — tích lũy qua thời gian, không bao giờ thụt lùi.
- **Ghép cặp trước tiên**: hiệu chuẩn chỉ đáng tin khi seraw↔SRT ghép đúng cặp.
  `auto_pair_seraw_to_srt` đo độ trùng n-gram và *từ chối* cặp không đủ tin cậy.

## Kiến trúc (tách tầng, Dependency Injection)

| Module | Trách nhiệm |
|--------|-------------|
| `metrics.py` | Hàm đo *thuần*: CER, Levenshtein, `score_subtitles`, `SubtitleScore.quality` |
| `ground_truth.py` | Parse SRT + ghép cặp seraw↔SRT bằng n-gram |
| `search_space.py` | `ParameterSpec` / `SearchSpace` (rời rạc hoá + thu hẹp biên) |
| `optimizer.py` | `RecursiveCoordinateDescentOptimizer` (đệ quy + cache) |
| `subtitle_evaluator.py` | Hàm mục tiêu build phụ đề (tiêm builder factory) |
| `roi_evaluator.py` | Hàm mục tiêu ROI (IoU khi có nhãn, proxy khi chưa) |
| `recursive_calibrator.py` | Điều phối + warm-start (lưu/nạp trạng thái) |
| `report.py` | Sinh báo cáo Markdown |
| `cli.py` | Điểm vào dòng lệnh |

Optimizer hoàn toàn *agnostic* — chỉ cần `objective(assignment) -> float`, nên
cùng một bộ tối ưu phục vụ cả hai chức năng.

## Cách dùng

```bash
# Hiệu chuẩn build phụ đề (dò nhanh trên cửa sổ thời gian hẹp)
PYTHONPATH=src:. python -m tools.calibration.cli \
    --mode subtitle --uploads /path/to/uploads \
    --state-dir tools/calibration/state --quick

# Hiệu chuẩn ROI offline từ seraw
PYTHONPATH=src:. python -m tools.calibration.cli \
    --mode roi --uploads /path/to/uploads --state-dir tools/calibration/state

# Hiệu chuẩn đầy đủ (không --quick) cả hai
PYTHONPATH=src:. python -m tools.calibration.cli --mode both --uploads /path/to/uploads
```

Báo cáo Markdown + trạng thái JSON được ghi vào `--state-dir`.

## Dữ liệu cần để chạy mỗi chu kỳ

- **Build phụ đề**: cặp `*_seraw.json` ↔ `*.srt` đã sửa tay theo video (ground-truth).
  Càng nhiều cặp đúng, hiệu chuẩn càng tổng quát.
- **ROI (chính xác)**: với mỗi video, toạ độ *dải phụ đề thật* (tỷ lệ Y top/bottom)
  để chấm IoU. Thiếu nhãn này, framework chỉ chạy được ở chế độ *proxy*
  (coverage × compactness), kém tin cậy hơn.
