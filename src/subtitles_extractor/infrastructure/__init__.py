"""Tầng Infrastructure — adapter cho thế giới bên ngoài.

Quy tắc:
    * Mỗi adapter hiện thực **đúng một** Protocol từ ``domain/ports``.
    * KHÔNG chứa logic nghiệp vụ — chỉ map giữa thư viện cụ thể và port.
    * Có thể thay thế bằng adapter khác mà không sửa application/domain.
"""
