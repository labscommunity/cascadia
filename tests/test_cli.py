"""CLI smoke tests — argparse wiring, simple subcommands."""

from __future__ import annotations

import os
from pathlib import Path

from tahoma.cli import _write_pid_file, cmd_engines, parse_addr


def test_parse_addr_with_host() -> None:
    assert parse_addr("10.0.0.1:9100") == ("10.0.0.1", 9100)


def test_parse_addr_default_host() -> None:
    assert parse_addr(":8000") == ("0.0.0.0", 8000)
    assert parse_addr(":8000", default_host="127.0.0.1") == ("127.0.0.1", 8000)


def test_parse_addr_handles_ipv4_with_port() -> None:
    assert parse_addr("192.168.86.250:9100") == ("192.168.86.250", 9100)


def test_pid_file_written_and_removed_on_exit(tmp_path: Path) -> None:
    """_write_pid_file writes our PID and registers atexit cleanup."""
    pid_path = tmp_path / "sub" / "tahoma.pid"
    _write_pid_file(pid_path)
    assert pid_path.exists()
    assert pid_path.read_text().strip() == str(os.getpid())
    # atexit cleanup is hard to test directly without forking; we just verify
    # the file has the expected contents and parent dir was created.
    assert pid_path.parent.is_dir()


def test_engines_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    import argparse
    rc = cmd_engines(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("pytorch", "ov-optimum", "ov-runtime", "ov-spec", "ov-dist-spec"):
        assert name in out
