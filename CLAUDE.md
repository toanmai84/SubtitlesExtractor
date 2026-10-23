# Quy tắc làm việc với dự án Subtitles Extractor

## Phát hiện lỗi thì khắc phục ngay (yêu cầu của chủ dự án)
- Trong lúc phát triển, kiểm thử hay đọc mã, hễ phát hiện lỗi (bug, crash, sai dữ liệu,
  treo, rò rỉ, test hỏng, phụ thuộc thiếu…) thì PHẢI đào sâu tới nguyên nhân gốc và khắc
  phục luôn — không chỉ ghi chú lại hay để "việc sau".
- Mỗi bản sửa: tái hiện lỗi trước (test thất bại trên mã cũ), sửa, thêm test chặn tái
  phát, chạy lại toàn bộ test. Ghi số liệu kiểm chứng thật vào CHANGELOG.
- Chỉ dừng lại hỏi khi cách sửa đòi quyết định sản phẩm (đổi hành vi người dùng thấy rõ,
  đánh đổi chất lượng/tốc độ, xoá tính năng).

## Giao bản
- Sau MỖI lần phát triển tính năng/cải tiến (không chỉ khi sửa lỗi): rà soát BUG và CRASH CHÍNH phần vừa
  viết (mypy, fuzz/ca biên, mô phỏng đa luồng nếu có khoá, đọc lại diff; việc lớn thì thêm một lượt
  rà độc lập bằng agent), khắc phục luôn rồi mới giao bản — không để người dùng phát hiện.
- Sau MỖI lần phát triển/sửa lỗi ứng dụng: chạy toàn bộ test, commit, rồi đóng gói zip mã nguồn
  (`git archive`, tên `SubtitlesExtractor_v<phiên_bản>_source.zip`) và GỬI cho chủ dự án.

## Quy ước
- Mỗi bản sửa tăng số bản (`pyproject.toml`, `raw_ocr_serializer.app_version`, `subtitles_extractor.__version__` —
  test `test_version_consistency_v323643` bắt lệch) và có mục
  CHANGELOG; chú thích trong mã gắn `[v3.23.xxx]`; test đặt tên `test_*_v323xxx.py`.
- Identifier tiếng Anh; docstring, log, giao diện tiếng Việt (chuỗi giao diện trong
  `src/subtitles_extractor/data/strings_vi.json` + `strings_en.json`).
- Chạy test: `QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests -q`.

## Cạm bẫy đã gặp
- Tiến trình chính giữ `sys.modules["torch"] = None` (bộ chặn torch) từ lúc khởi động đến khi PaddleOCR nạp xong; trong lúc đó
  `import scipy.stats|signal`/`sklearn.cluster` lần đầu chết với `'NoneType' object has no attribute 'Tensor'`. Đổi import eager → lazy
  (hoặc đổi thứ tự nạp) phải kiểm bằng `test_paddle_import_v323646.py` (v3.23.643 từng làm hỏng OCR trên exe).

## Dữ liệu mẫu
- Bộ phim 86 tập hardsub tiếng Trung giản thể (720x1280 HEVC; tập 56-58, 63-65 là 30 fps,
  còn lại 25 fps). Dùng để kiểm tra chất lượng trích xuất, dịch, lồng tiếng, nối cả bộ.
