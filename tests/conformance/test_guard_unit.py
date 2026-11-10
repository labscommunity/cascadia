"""Unit tests for the backing-dir guard in test_elastic.py (no model, no GPU).

Run: python3 -m pytest -q tests/conformance/test_guard_unit.py
"""
import collections
import importlib.util
import os
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "conformance_elastic", pathlib.Path(__file__).with_name("test_elastic.py"))
te = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(te)

Usage = collections.namedtuple("Usage", "total used free")
GIB = 1 << 30


@pytest.fixture
def two_disk_candidates(tmp_path, monkeypatch):
    """/var/tmp and ~/.cache both exist, are disk-typed and have 20 GiB free."""
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(te, "fstype", lambda p: "ext4")
    monkeypatch.setattr(te.shutil, "disk_usage", lambda p: Usage(0, 0, 20 * GIB))
    real_isdir = os.path.isdir
    monkeypatch.setattr(te.os.path, "isdir",
                        lambda p: True if p == "/var/tmp" else real_isdir(p))
    return str(home / ".cache")


def test_rejects_candidates_that_refuse_o_tmpfile(two_disk_candidates, monkeypatch):
    # Writable dirs that reject O_TMPFILE (NFS/FUSE): the shim stays on tmpfs,
    # so the guard must not report an escape.
    def no_tmpfile(path, flags, mode=0o777):
        if flags & getattr(os, "O_TMPFILE", 0):
            raise OSError(95, "Operation not supported")
        return os.open(path, flags, mode)
    monkeypatch.setattr(te.os, "open", no_tmpfile)
    assert te.escape_target("/dev/shm") is None


def test_rejects_candidates_that_refuse_ftruncate(two_disk_candidates, monkeypatch):
    if not hasattr(os, "O_TMPFILE"):
        pytest.skip("needs O_TMPFILE")
    monkeypatch.setattr(te, "dir_usable", lambda p: False)
    assert te.escape_target("/dev/shm") is None


def test_dir_usable_probe_on_a_real_dir(tmp_path):
    if not hasattr(os, "O_TMPFILE"):
        pytest.skip("needs O_TMPFILE")
    assert te.dir_usable(str(tmp_path)) in (True, False)  # fs-dependent, must not raise
    assert te.dir_usable(str(tmp_path / "missing")) is False


def test_nondefault_pool_needs_more_space(two_disk_candidates, monkeypatch):
    monkeypatch.setattr(te, "dir_usable", lambda p: True)
    # 20 GiB free: enough for the default 8 GiB pool + 1 GiB, not for 16 GiB + 1.
    assert te.escape_target("/dev/shm", te.SHIM_POOL_MB_DEFAULT) == "/var/tmp"
    assert te.escape_target("/dev/shm", te.shim_pool_mb({"ELASTIC_POOL_MB": "24576"})) is None


@pytest.mark.parametrize("value, want", [
    (None, 8192), ("16384", 16384), ("0", 0), ("-5", 8192),
    ("abc", 0), ("12mb", 12), (str(1 << 30), 1 << 20),
])
def test_shim_pool_mb_mirrors_the_shim(value, want):
    env = {} if value is None else {"ELASTIC_POOL_MB": value}
    assert te.shim_pool_mb(env) == want
