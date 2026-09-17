import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _cuda_capability() -> tuple[int, int]:
    if not _has_cuda():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA")
    config.addinivalue_line("markers", "sm100: requires SM100/SM110")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    have_cuda = _has_cuda()
    cc = _cuda_capability()
    is_sm100 = cc[0] in (10, 11)
    for item in items:
        if "cuda" in item.keywords and not have_cuda:
            item.add_marker(pytest.mark.skip(reason="no CUDA device"))
        if "sm100" in item.keywords and not is_sm100:
            item.add_marker(
                pytest.mark.skip(reason=f"needs SM100/SM110, device cc={cc}")
            )
