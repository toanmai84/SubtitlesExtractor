"""Tầng Application — chứa use case orchestration.

Quy tắc: phụ thuộc *vào* domain (entities, ports), KHÔNG phụ thuộc
infrastructure hay presentation. Tầng này có thể test 100% offline
bằng cách inject các Protocol mock.
"""
