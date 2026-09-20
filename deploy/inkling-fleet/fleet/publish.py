#!/usr/bin/env python3
"""Publish the fleet's files from rank 0: write a signed manifest.json next to them.

    python3 publish.py [folder, default ~/inkling-files] [--key ~/.inkling-fleet.key] [--new-key]

The folder is what the file server on rank 0 serves (python3 -m http.server 8088). Put the new run.sh,
cascadia, beacon.py, status.sh, updater.py or fleet-overrides.env in it and run this; every box's updater
installs what changed within about 20 s. `--new-key` creates the fleet key (once, before enrolling boxes).
"""
import argparse, hashlib, hmac, json, os, secrets, sys, time

NAMES = ["run.sh", "cascadia", "fleet-overrides.env", "beacon.py", "status.sh", "updater.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=os.path.expanduser("~/inkling-files"))
    ap.add_argument("--key", default=os.path.expanduser("~/.inkling-fleet.key"))
    ap.add_argument("--new-key", action="store_true")
    a = ap.parse_args()
    if a.new_key:
        if os.path.exists(a.key):
            sys.exit("%s already exists: boxes enrolled with it would stop accepting updates if it changed" % a.key)
        fd = os.open(a.key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, secrets.token_hex(32).encode() + b"\n"); os.close(fd)
        print("created", a.key)
    key = open(a.key).read().strip().encode()
    files = {}
    for n in NAMES:
        p = os.path.join(a.folder, n)
        if os.path.isfile(p):
            data = open(p, "rb").read()
            files[n] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    version = int(time.time())
    canonical = json.dumps({"version": version, "files": files}, sort_keys=True, separators=(",", ":")).encode()
    manifest = {"version": version, "files": files, "hmac": hmac.new(key, canonical, hashlib.sha256).hexdigest()}
    tmp = os.path.join(a.folder, "manifest.json.tmp")
    json.dump(manifest, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(a.folder, "manifest.json"))
    print("published version %d: %s" % (version, ", ".join("%s (%d B)" % (n, m["size"]) for n, m in files.items())))


if __name__ == "__main__":
    main()
