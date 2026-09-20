#!/usr/bin/env python3
"""Fleet updater: every box pulls the fleet's scripts and binary from rank 0.

Runs as cascadia-inkling-updater.service (root). Every ~15 s it fetches
http://<fleet>-rank-0:<port>/manifest.json (the name is kept current by the
beacon), checks its signature, and installs the files that differ from what is
on this box, then restarts the services that use them. Rank 0 publishes with
publish.py; nothing is pushed and no passwords or SSH are involved.

Safety: the manifest is signed (HMAC-SHA256) with the fleet key in
<prefix>/fleet.key, which only the fleet's boxes hold, so another machine on the
LAN that claims to be rank 0 cannot make the boxes run anything. Each file is
checked against the signed sha256 before it replaces the old one (atomically),
only the names in FILES are ever written, and a manifest older than the last
one applied is refused (no roll-back by replay).
"""
import argparse
import hashlib
import hmac
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

# name in the manifest -> (file name under the prefix, mode, what to restart)
FILES = {
    "run.sh": ("run.sh", 0o755, "worker"),
    "cascadia": ("cascadia", 0o755, "worker"),
    "fleet-overrides.env": ("fleet-overrides.env", 0o644, "worker"),
    "beacon.py": ("beacon.py", 0o755, "beacon"),
    "status.sh": ("status.sh", 0o755, None),
    "updater.py": ("updater.py", 0o755, "self"),
}
UNITS = {"worker": "cascadia-inkling.service", "beacon": "cascadia-inkling-beacon.service"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def canonical(version, files):
    return json.dumps({"version": version, "files": files}, sort_keys=True, separators=(",", ":")).encode()


def sign(key, version, files):
    return hmac.new(key, canonical(version, files), hashlib.sha256).hexdigest()


def sha256_of(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def fetch(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def write_state(path, **kw):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(kw, f)
        os.replace(tmp, path)
    except OSError:
        pass


def check_once(a, key, applied):
    """Returns (version now applied, units to restart, restart self?)."""
    base = "http://%s-rank-0:%d" % (a.fleet, a.port)
    m = json.loads(fetch(base + "/manifest.json").decode())
    version, files, sig = int(m["version"]), m["files"], str(m.get("hmac", ""))
    if not hmac.compare_digest(sig, sign(key, version, files)):
        raise ValueError("manifest signature does not match this fleet's key: ignored")
    if version < applied:
        raise ValueError("manifest version %d is older than %d already applied: ignored" % (version, applied))
    restart, restart_self = set(), False
    for name, meta in sorted(files.items()):
        if name not in FILES:
            continue
        dest_name, mode, what = FILES[name]
        dest = os.path.join(a.prefix, dest_name)
        if sha256_of(dest) == meta["sha256"]:
            continue
        data = fetch(base + "/" + name, timeout=120)
        if len(data) != int(meta["size"]) or hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError("%s: download does not match the signed checksum: nothing installed" % name)
        tmp = dest + ".new"
        with open(tmp, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, dest)
        log("installed %s (%d bytes) from fleet files version %d" % (name, len(data), version))
        if what == "self":
            restart_self = True
        elif what:
            restart.add(what)
    return version, restart, restart_self


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="/opt/cascadia-inkling")
    ap.add_argument("--fleet", default="inkling")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--state", default="/run/cascadia-inkling/update.json")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    keyfile = os.path.join(a.prefix, "fleet.key")
    persist = os.path.join(a.prefix, "update-state.json")
    try:
        applied = int(json.load(open(persist)).get("version", 0))
    except (OSError, ValueError):
        applied = 0
    log("updater: fleet %s, files from http://%s-rank-0:%d, last applied version %d" % (a.fleet, a.fleet, a.port, applied))
    last_err = ""
    while True:
        try:
            key = open(keyfile).read().strip().encode()
            if len(key) < 32:
                raise ValueError("%s is missing or too short: this box is not enrolled" % keyfile)
            version, restart, restart_self = check_once(a, key, applied)
            if version != applied or restart or restart_self:
                applied = version
                with open(persist + ".tmp", "w") as f:
                    json.dump({"version": applied, "time": time.time()}, f)
                os.replace(persist + ".tmp", persist)
            write_state(a.state, version=applied, ok=True, time=time.time(), error="")
            last_err = ""
            for what in ("beacon", "worker"):
                if what in restart:
                    log("restarting %s" % UNITS[what])
                    subprocess.call(["systemctl", "restart", UNITS[what]])
            if restart_self:
                log("the updater itself changed: exiting so systemd starts the new one")
                return 0
        except Exception as e:  # never die: the next round may work (rank 0 not up yet, LAN hiccup, ...)
            err = "%s: %s" % (type(e).__name__, e)
            if err != last_err:
                log("no update this round: " + err)
                last_err = err
            write_state(a.state, version=applied, ok=False, time=time.time(), error=err[:200])
        if a.once:
            return 0 if not last_err else 1
        time.sleep(a.interval + random.uniform(0, 5))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
