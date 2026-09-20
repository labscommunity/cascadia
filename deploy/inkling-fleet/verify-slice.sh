#!/bin/bash
# Check this box's Inkling slice byte for byte against the SSD it was copied from.
#     sudo ./verify-slice.sh /media/$USER/<ssd> [install folder, default /opt/cascadia-inkling]
# Rebuilds the list of files this rank must hold (the installer's rule), then for each one: present, same size,
# same bytes. Also reports files that should not be there. Read-only. A few minutes (it reads the slice twice).
set -u
SSD="${1:?usage: $0 /path/to/ssd [install folder]}"; PREFIX="${2:-/opt/cascadia-inkling}"
[ -f "$SSD/inkling/out/manifest.json" ] || { echo "no inkling/out/manifest.json under $SSD - is this the SSD's top folder?"; exit 1; }
[ -f "$PREFIX/model/rank.json" ] || { echo "no $PREFIX/model/rank.json - is a rank installed under $PREFIX?"; exit 1; }
exec python3 - "$SSD/inkling/out" "$PREFIX/model" <<'PY'
import json, os, sys, time
src, dst = sys.argv[1], sys.argv[2]
rj = json.load(open(os.path.join(dst, "rank.json")))
rank, total, lo, hi = rj["rank"], rj["total"], rj["layer_start"], rj["layer_end"]
n = int(json.load(open(os.path.join(src, "manifest.json")))["num_layers"])
base, rem = divmod(n, total); want_lo = rank * base + min(rank, rem); want_hi = want_lo + base + (1 if rank < rem else 0)
print("this box: rank %d of %d, layers [%d,%d) of %d" % (rank, total, lo, hi, n))
problems = []
if (lo, hi) != (want_lo, want_hi):
    problems.append("rank.json says layers [%d,%d) but rank %d of %d must hold [%d,%d)" % (lo, hi, rank, total, want_lo, want_hi))
files = ["manifest.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "source_config.json"]
for l in range(lo, hi):
    files.append("shells/layer_%02d.safetensors" % l)
    files += ["experts/layer_%02d/%s" % (l, x) for x in sorted(os.listdir(os.path.join(src, "experts", "layer_%02d" % l)))]
    adir = os.path.join(src, "attn_ov", "layer_%02d" % l)
    for root, _, names in os.walk(adir):
        files += [os.path.relpath(os.path.join(root, x), src) for x in sorted(names)]
if rank == 0: files.append("embed.safetensors")
if rank == total - 1:
    files.append("head.safetensors")
    if os.path.isdir(os.path.join(src, "head_ov")): files += ["head_ov/" + x for x in sorted(os.listdir(os.path.join(src, "head_ov")))]
files = [f for f in files if os.path.exists(os.path.join(src, f))]
total_bytes = sum(os.path.getsize(os.path.join(src, f)) for f in files)
print("expected: %d files, %.1f GiB" % (len(files), total_bytes / 2**30), flush=True)
t0 = time.time(); done = 0; step = 0; CH = 8 << 20
for f in files:
    s, d = os.path.join(src, f), os.path.join(dst, f)
    size = os.path.getsize(s)
    if not os.path.exists(d): problems.append("MISSING  " + f)
    elif os.path.getsize(d) != size: problems.append("SIZE     %s (%d on the box, %d on the SSD)" % (f, os.path.getsize(d), size))
    else:
        with open(s, "rb") as a, open(d, "rb") as b:
            while True:
                x, y = a.read(CH), b.read(CH)
                if x != y: problems.append("CONTENT  " + f); break
                if not x: break
    done += size
    if total_bytes and int(done * 10 / total_bytes) > step:
        step = int(done * 10 / total_bytes); print("  %3d%%  %.0f s" % (step * 10, time.time() - t0), flush=True)
expected = set(files) | {"rank.json"}
extra = []
for root, dirs, names in os.walk(dst):
    dirs[:] = [x for x in dirs if not (root == dst and x == "moe_ov")]   # fused layers generated on the box
    extra += [os.path.relpath(os.path.join(root, x), dst) for x in names if os.path.relpath(os.path.join(root, x), dst) not in expected]
if extra: print("note: %d file(s) on the box that this rank does not need, e.g. %s (harmless; left by an install with another rank or fleet size)" % (len(extra), extra[0]))
fused = os.path.join(dst, "moe_ov")
if os.path.isdir(fused): print("fused iGPU layers generated on this box: %s" % " ".join(sorted(os.listdir(fused))))
if problems:
    print("\nPROBLEMS (%d):" % len(problems)); [print("  " + p) for p in problems[:20]]
    print("fix: run the installer again with the same rank; it re-copies files whose size is wrong. For CONTENT, delete the listed files first.")
    sys.exit(1)
print("\nOK: all %d files are present and identical to the SSD (%.1f GiB compared in %.0f s)" % (len(files), total_bytes / 2**30, time.time() - t0))
PY
