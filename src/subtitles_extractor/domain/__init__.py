"""Tầng nghiệp vụ thuần (Domain).

Quy tắc: chỉ phụ thuộc thư viện chuẩn của Python. KHÔNG import:
    * PyQt6, qfluentwidgets
    * paddle, paddleocr
    * cv2, av, decord
    * numpy (ngoại trừ Polygon trong type-hint)
"""
