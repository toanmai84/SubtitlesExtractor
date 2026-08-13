"""Bộ kiểm thử tích hợp — mô phỏng các luồng nghiệp vụ hoàn chỉnh.

Khác với unit test (kiểm từng đơn vị cô lập), các test ở đây nối nhiều use case + service
+ store thật với nhau, chỉ giả lập (fake) ở RANH GIỚI NGOÀI (Gemini/OCR/TTS) — những thành
phần cần mạng hoặc thư viện nặng không có trong môi trường CI.
"""
