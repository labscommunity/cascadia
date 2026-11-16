"""Lock-time check that the PyTorch CPU index cannot shadow a locked wheel (stdlib only).

The CLI syncs with `--index-strategy unsafe-best-match` and that index as an extra, so for every
pin uv looks at both indexes. `==1.2` also matches a local version such as `1.2+cpu`, so a wheel
on the torch index that matches a pin but carries a hash the lock does not list (a `+cpu` build of
a package locked from PyPI) can win, and the sync then fails its hash check. Such a package must be
sourced from the torch index in pyproject.toml. Usage: python3 -I index_guard.py requirements.txt
"""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_LINK = re.compile(r'href="([^"]+\.whl)#sha256=([0-9a-f]{64})"')


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(text: str) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """The `name==version` pins in order, and every hash locked for each package."""
    pins: list[tuple[str, str]] = []
    hashes: dict[str, set[str]] = {}
    name = None
    for line in text.splitlines():
        m = _PIN.match(line)
        if m:
            name = normalize(m.group(1))
            if (name, m.group(2)) not in pins:
                pins.append((name, m.group(2)))
            hashes.setdefault(name, set())
        elif not line.startswith(" "):
            name = None
        if name:
            hashes[name].update(_HASH.findall(line))
    return pins, hashes


def wheel_links(html: str) -> list[tuple[str, str]]:
    """(filename, sha256) of every wheel on a simple-index page."""
    return [
        (urllib.parse.unquote(href.rsplit("/", 1)[-1]), sha) for href, sha in _LINK.findall(html)
    ]


def cp312_compatible(filename: str) -> bool:
    parts = filename[: -len(".whl")].split("-")
    py, abi = parts[-3].split("."), parts[-2]
    if {"cp312", "py3", "py312"} & set(py):
        return True
    return abi == "abi3" and any(re.fullmatch(r"cp3(\d+)", t) and int(t[3:]) <= 12 for t in py)


def matches(pin: str, version: str) -> bool:
    """PEP 440 `==pin` without a wildcard: a pin with no local version matches any local one."""
    return version == pin or ("+" not in pin and version.startswith(pin + "+"))


def shadows(
    pins: list[tuple[str, str]], hashes: dict[str, set[str]], links: dict[str, list[tuple[str, str]]]
) -> list[str]:
    """Each torch-index wheel a Python 3.12 sync could pick for a pin whose hash is not locked."""
    found = []
    for name, pin in pins:
        for filename, sha in links.get(name, []):
            version = filename.split("-")[1]
            if cp312_compatible(filename) and matches(pin, version) and sha not in hashes[name]:
                found.append(f"{name}=={pin}: {filename} (sha256:{sha}) is not in the lock")
    return found


def fetch(name: str) -> list[tuple[str, str]]:
    req = urllib.request.Request(f"{TORCH_CPU_INDEX}/{name}/", headers={"Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return wheel_links(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return []
        raise


def main(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        pins, hashes = parse_lock(f.read())
    names = sorted({n for n, _ in pins})
    with ThreadPoolExecutor(8) as pool:
        links = dict(zip(names, pool.map(fetch, names)))
    found = shadows(pins, hashes, links)
    for line in found:
        print(f"index_guard: {line}", file=sys.stderr)
    if found:
        print(
            "index_guard: source these packages from the pytorch-cpu index in pyproject.toml "
            "([tool.uv.sources], with a marker for the platforms it carries) and relock",
            file=sys.stderr,
        )
        return 1
    print(f"index_guard: {len(pins)} pins, none shadowed by {TORCH_CPU_INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
