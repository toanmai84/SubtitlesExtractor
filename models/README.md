# models/ — Kho model tập trung

Thư mục này gom **mọi model mà ứng dụng cần tải về**, song song với `vendor/` (binary
native). Mục tiêu: tập trung một chỗ, dễ thêm/bớt/cập nhật, gốc dự án sạch, và `.spec`
chỉ cần nhúng nguyên cây `models/`.

## Cấu trúc

```
models/
  paddle/       official_models/<tên model>/...      — PaddleOCR (detection/recognition)
  huggingface/  hub/models--<org>--<repo>/...        — VieNeu-TTS (ONNX), fastembed
  README.md     (file này)
```

## Cách app tìm model (runtime)

`infrastructure/model_store.py` phân giải gốc `models` theo thứ tự:

1. Biến môi trường `SUBEXT_MODELS_DIR` (trỏ gốc kho tùy ý — escape hatch).
2. Bản đóng gói: `sys._MEIPASS/models` (do `.spec` nhúng).
3. Chạy nguồn: `<gốc dự án>/models` (cạnh `main.py`).

Sau đó, `main()` gọi `configure_all_model_stores()` **trước mọi import nặng** để set:

| Thư mục | Biến môi trường | Thư viện dùng |
|---------|-----------------|---------------|
| `models/paddle` | `PADDLE_PDX_CACHE_HOME` | PaddleOCR / PaddleX |
| `models/huggingface` | `HF_HOME` | VieNeu-TTS, fastembed (qua `huggingface_hub`) |

**Nguyên tắc an toàn:** chỉ set biến khi model thực sự có sẵn; tôn trọng biến môi
trường người dùng đã đặt; **không** bật `HF_HUB_OFFLINE` nên model chưa có vẫn tải
được bình thường.

## Prefetch (chuẩn bị model trước khi build)

```bash
# Model PaddleOCR — phiên bản mặc định × toàn bộ ngôn ngữ UI
python tools/prefetch_ocr_models.py

# Model HuggingFace — mặc định VieNeu-TTS v3 Turbo
python tools/prefetch_hf_models.py

# Tùy chọn: thêm model embedding cho Translation Memory ngữ nghĩa
python tools/prefetch_hf_models.py --repos BAAI/bge-small-zh-v1.5
```

`build_windows.bat` đã tự chạy cả hai bước này trước khi gọi PyInstaller.

## Lưu ý git

Model rất lớn (hàng trăm MB → GB) nên được loại khỏi git qua `.gitignore` (chỉ giữ
`README.md` và cấu trúc thư mục). Khi build ở máy khác, chạy lại các lệnh prefetch ở trên.
