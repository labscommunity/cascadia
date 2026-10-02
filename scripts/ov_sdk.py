#!/usr/bin/env python3
"""Resolve and fetch an OpenVINO GenAI SDK archive by version.

The one place that knows how Intel lays out
https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/ —
used by release.yml / ci.yml (so the bundled runtime version is a single
string, not two hand-copied URLs) and by people building from source who
want a specific SDK, including a beta:

    python3 scripts/ov_sdk.py resolve 2026.4.1.0 --os linux --dist ubuntu24
    python3 scripts/ov_sdk.py fetch   2026.5.0.0beta1 --dest ~/openvino
    export INTEL_OPENVINO_DIR="$(python3 scripts/ov_sdk.py fetch 2026.5.0.0beta1 --dest ~/openvino)"

Version strings are Intel's archive versions, exactly as they appear in the
file names: ``2026.4.1.0`` (stable), ``2026.5.0.0beta1`` (beta, published
under ``packages/beta/``), ``2026.5.0.0.dev20260925`` (nightly, under
``packages/nightly/``). Stable directories drop trailing ``.0`` components
(``2026.4.1.0`` lives in ``packages/2026.4.1/``), but not consistently —
``2026.1.2.0`` keeps its full name — so every candidate URL is probed.

The storage host answers HTTP 200 with a small HTML placeholder for paths
that do NOT exist, so a plain GET/HEAD proves nothing. A ranged request is
the test: a real archive answers 206 (Partial Content).

Standard library only (no pip), Python 3.8+.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile

BASE = "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages"
MARKER = ".cascadia-ov-sdk"  # written into an extracted SDK root: "<version> <os> <dist>"

VERSION_RE = re.compile(r"^\d{4}\.\d+(\.\d+){0,2}([A-Za-z0-9.]*)$")


def channel(version: str) -> str:
    """'stable', 'beta' or 'nightly' from the version string alone."""
    v = version.lower()
    if ".dev" in v:
        return "nightly"
    if "beta" in v or "rc" in v:
        return "beta"
    return "stable"


def stable_dir_names(version: str) -> list[str]:
    """Directory names a stable archive may live under, most likely first.

    Intel trims trailing '.0' components for most releases (2026.4.1.0 ->
    2026.4.1, 2026.4.0.0 -> 2026.4) but keeps the full string for a few
    (2026.1.2.0). Always keep at least major.minor.
    """
    parts = version.split(".")
    names = []
    trimmed = list(parts)
    while len(trimmed) > 2 and trimmed[-1] == "0":
        trimmed.pop()
    names.append(".".join(trimmed))
    if version not in names:
        names.append(version)
    return names


def archive_name(version: str, os_name: str, dist: str | None) -> str:
    if os_name == "linux":
        if not dist:
            raise ValueError("linux needs --dist (ubuntu22 | ubuntu24 | ubuntu26 | rhel8 | rhel9)")
        return f"openvino_genai_{dist}_{version}_x86_64.tar.gz"
    if os_name == "windows":
        return f"openvino_genai_windows_{version}_x86_64.zip"
    if os_name == "macos":
        return f"openvino_genai_macos_12_6_{version}_arm64.tar.gz"
    raise ValueError(f"unknown os {os_name!r} (linux | windows | macos)")


def candidate_urls(version: str, os_name: str, dist: str | None) -> list[str]:
    """Every URL the archive could be at, most likely first. Pure; no I/O."""
    if not VERSION_RE.match(version):
        raise ValueError(
            f"{version!r} does not look like an OpenVINO GenAI archive version "
            "(e.g. 2026.4.1.0, 2026.5.0.0beta1, 2026.5.0.0.dev20260925)"
        )
    name = archive_name(version, os_name, dist)
    ch = channel(version)
    if ch == "beta":
        dirs = [f"beta/{version}"]
    elif ch == "nightly":
        dirs = [f"nightly/{version}"]
    else:
        dirs = stable_dir_names(version)
    return [f"{BASE}/{d}/{os_name}/{name}" for d in dirs]


def exists(url: str, timeout: float = 30.0) -> bool:
    """True only for a real archive: a ranged GET must answer 206."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "cascadia-ov-sdk"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 206
    except urllib.error.HTTPError as e:
        if e.code in (404, 416):
            return False
        raise


def resolve(version: str, os_name: str, dist: str | None) -> str:
    cands = candidate_urls(version, os_name, dist)
    for u in cands:
        if exists(u):
            return u
    raise SystemExit(
        f"ov_sdk: no OpenVINO GenAI {version} archive for {os_name}"
        + (f"/{dist}" if dist else "")
        + " (probed with a ranged GET, none answered 206):\n  "
        + "\n  ".join(cands)
        + f"\nBrowse {BASE}/ for what Intel has published."
    )


def default_os() -> str:
    s = platform.system().lower()
    if s.startswith("linux"):
        return "linux"
    if s.startswith("win"):
        return "windows"
    if s.startswith("darwin"):
        return "macos"
    raise SystemExit(f"ov_sdk: unsupported host OS {s!r}; pass --os")


def default_dist() -> str | None:
    """ubuntu22 / ubuntu24 / ubuntu26 from /etc/os-release, else None."""
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            kv = dict(
                line.rstrip("\n").split("=", 1) for line in f if "=" in line and not line.startswith("#")
            )
    except OSError:
        return None
    ident = kv.get("ID", "").strip('"')
    ver = kv.get("VERSION_ID", "").strip('"')
    major = int(ver.split(".")[0]) if ver.split(".")[0].isdigit() else 0
    if ident == "ubuntu":
        # No archive for 23.x/25.x; use the nearest LTS at or below.
        if major >= 26:
            return "ubuntu26"
        if major >= 24:
            return "ubuntu24"
        return "ubuntu22"
    if ident in ("rhel", "rocky", "almalinux", "centos"):
        return "rhel9" if major >= 9 else "rhel8"
    return None


def _download(url: str, dest_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "cascadia-ov-sdk"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_path, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)


def parse_sha256_sidecar(text: str, archive_basename: str) -> tuple[str | None, str | None]:
    """(hash, reason-it-was-ignored) from a '<archive>.sha256' sidecar's text.

    The sidecar is sha256sum output: "<hex>  <file name>". It is only a
    checksum of OUR download when the file name it carries is the archive we
    fetched. Intel's beta sidecars have been seen naming a different file
    (2026.5.0.0beta1's say "…2026.5.0.0.dev20260917…" with a hash that matches
    neither), so a name mismatch means "no usable checksum", not "corrupt".
    A placeholder HTML page (missing sidecar) has no 64-hex token at all.
    """
    tok = text.split()[0] if text.strip() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", tok):
        return None, "no published .sha256 for this archive"
    rest = text.strip().split(None, 1)
    name = rest[1].strip().lstrip("*") if len(rest) > 1 else ""
    if name and os.path.basename(name.replace("\\", "/")) != archive_basename:
        return None, f"published .sha256 names a different file ({name}); not used"
    return tok.lower(), None


def _expected_sha256(url: str) -> tuple[str | None, str | None]:
    """Intel publishes '<archive>.sha256' beside every archive; use it when it is really ours."""
    try:
        req = urllib.request.Request(url + ".sha256", headers={"User-Agent": "cascadia-ov-sdk"})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                return None, "no published .sha256 for this archive"
            text = r.read(4096).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return None, f"could not read the published .sha256 ({e})"
    return parse_sha256_sidecar(text, url.rsplit("/", 1)[1])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_stripped(archive: str, dest: str) -> None:
    """Extract into dest, dropping the single top-level SDK directory."""
    tmp = tempfile.mkdtemp(prefix="ov_sdk_", dir=os.path.dirname(os.path.abspath(dest)) or None)
    try:
        if archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(tmp)
        else:
            with tarfile.open(archive, "r:gz") as t:
                # Python >= 3.12 warns without a filter; 'data' rejects absolute
                # paths and '..' members, which an SDK archive never needs.
                if hasattr(tarfile, "data_filter"):
                    t.extractall(tmp, filter="data")
                else:
                    t.extractall(tmp)
        entries = [e for e in os.listdir(tmp) if not e.startswith(".")]
        if len(entries) != 1 or not os.path.isdir(os.path.join(tmp, entries[0])):
            raise SystemExit(f"ov_sdk: expected one top-level directory in {archive}, found {entries}")
        root = os.path.join(tmp, entries[0])
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        shutil.move(root, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(version: str, os_name: str, dist: str | None, dest: str, force: bool) -> str:
    """Download + extract once; print nothing but the SDK root on stdout."""
    dest = os.path.abspath(dest)
    stamp = f"{version} {os_name} {dist or '-'}"
    marker = os.path.join(dest, MARKER)
    if not force and os.path.isdir(os.path.join(dest, "runtime", "include")) and os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            if f.read().strip() == stamp:
                print(f"ov_sdk: {dest} already holds OpenVINO GenAI {version} ({os_name}/{dist or '-'})", file=sys.stderr)
                return dest
    url = resolve(version, os_name, dist)
    print(f"ov_sdk: fetching {url}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="ov_sdk_dl_") as td:
        archive = os.path.join(td, url.rsplit("/", 1)[1])
        _download(url, archive)
        want, why_not = _expected_sha256(url)
        if want:
            got = _sha256(archive)
            if got != want:
                raise SystemExit(f"ov_sdk: sha256 mismatch for {url}\n  published {want}\n  downloaded {got}")
            print("ov_sdk: sha256 verified", file=sys.stderr)
        else:
            print(f"ov_sdk: {why_not}; skipping checksum", file=sys.stderr)
        _extract_stripped(archive, dest)
    if not os.path.isdir(os.path.join(dest, "runtime", "include")):
        raise SystemExit(f"ov_sdk: {dest} has no runtime/include/ after extraction — not an SDK archive?")
    with open(marker, "w", encoding="utf-8") as f:
        f.write(stamp + "\n")
    print(f"ov_sdk: OpenVINO GenAI {version} extracted to {dest}", file=sys.stderr)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("version", help="archive version, e.g. 2026.4.1.0 or 2026.5.0.0beta1")
        sp.add_argument("--os", dest="os_name", choices=["linux", "windows", "macos"], default=None,
                        help="target OS (default: this host)")
        sp.add_argument("--dist", default=None,
                        help="linux distro tag in the archive name: ubuntu22 | ubuntu24 | ubuntu26 | rhel8 | rhel9 "
                             "(default: from /etc/os-release)")

    r = sub.add_parser("resolve", help="print the archive URL (after proving it exists)")
    common(r)
    c = sub.add_parser("candidates", help="print every candidate URL without touching the network")
    common(c)
    f = sub.add_parser("fetch", help="download + extract; print the SDK root (use as INTEL_OPENVINO_DIR)")
    common(f)
    f.add_argument("--dest", required=True, help="directory to extract the SDK root into (created/replaced)")
    f.add_argument("--force", action="store_true", help="re-download even if --dest already holds this version")

    a = p.parse_args(argv)
    os_name = a.os_name or default_os()
    dist = a.dist
    if os_name == "linux" and not dist:
        dist = default_dist() if a.os_name is None or platform.system().lower().startswith("linux") else None
        if not dist:
            p.error("--dist is required for --os linux when the host is not Ubuntu/RHEL")
    try:
        if a.cmd == "candidates":
            print("\n".join(candidate_urls(a.version, os_name, dist)))
        elif a.cmd == "resolve":
            print(resolve(a.version, os_name, dist))
        elif a.cmd == "fetch":
            print(fetch(a.version, os_name, dist, a.dest, a.force))
    except ValueError as e:
        p.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
