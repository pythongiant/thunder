"""Fast-launch key: every launch-time constexpr must be in the key (CPU)."""

from __future__ import annotations

import pytest

pytest.importorskip("cutlass")

from thunder_vllm.attention.cute_kernel import _fast_key  # noqa: E402


def _key(**over):
    kw = dict(
        kernel_cfg=(128, 4, 4, 4, False, 64, 64, 128),
        shapes=((1, 32, 128), (256, 16, 8, 64)),
        num_reqs=1, max_query_len=1, num_splits=1,
        debug=False, split_mode=0, gqa_mode=0, gqa_pack=False,
        onepass=True, reg_rescale=True, causal_bound=True, indirect=True,
    )
    kw.update(over)
    return _fast_key(**kw)


def test_same_launch_same_key():
    assert _key() == _key()


def test_gqa_pack_changes_key():
    assert _key(gqa_pack=True) != _key(gqa_pack=False)


def test_indirect_changes_key():
    assert _key(indirect=True) != _key(indirect=False)


def test_each_scalar_changes_key():
    base = _key()
    for name, alt in (("num_reqs", 2), ("max_query_len", 2), ("num_splits", 2),
                      ("debug", True), ("split_mode", 1), ("gqa_mode", 1),
                      ("onepass", False), ("reg_rescale", False),
                      ("causal_bound", False)):
        assert _key(**{name: alt}) != base, name

def test_kernel_cfg_changes_key():
    """The tile is a launch-time constexpr, and the tile policy now moves it
    between schedules (decode 64 -> 32 rows, prefill KV 64 -> 16 wide). A key that
    ignored it would let one schedule's launch serve another -- the aliasing
    failure mode 4 records."""
    assert _key(kernel_cfg=(128, 4, 4, 4, False, 32, 32, 64)) != _key()
    assert _key(kernel_cfg=(128, 4, 4, 4, False, 64, 16, 128)) != _key()
    assert _key(kernel_cfg=(128, 4, 4, 4, True, 64, 64, 128)) != _key()


def test_shapes_change_key():
    """CuTeDSL's jit_cache is keyed per shape, so the fast-launch key must be too:
    handing one shape's compiled function to another is unsafe."""
    assert _key(shapes=((1, 32, 128), (256, 16, 8, 64), (1,))) != _key()
    assert _key(shapes=((4096, 32, 128), (1024, 16, 8, 64))) != _key()
