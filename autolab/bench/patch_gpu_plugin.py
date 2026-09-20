import hashlib, os, shutil, sys
# OpenVINO 2026.3.1 GPU plugin: the fused-MoE decode kernels index an expert's weights with a 32-bit product
# (expert_id * expert_wei_size): 9,437,184 bytes x id >= 228 passes INT_MAX and reads out of bounds. Same-length
# text patch of the embedded OpenCL source, 4 sites (gate/up and down kernels, u4 and u8 branches): int -> long.
lib = sys.argv[1]
ORIG = sys.argv[2]
A = b"const int expert_wei_size=INTERMEDIATE_SIZE * HIDDEN_SIZE"
B = b"const long expert_wei_size=INTERMEDIATE_SIZE *HIDDEN_SIZE"
assert len(A) == len(B)
data = open(lib, "rb").read()
sha = hashlib.sha256(data).hexdigest()
if data.count(B) == 4 and data.count(A) == 0:
    print("already patched", sha); sys.exit(0)
if sha != ORIG:
    print("unknown library", sha); sys.exit(3)
if data.count(A) != 4:
    print("unexpected site count", data.count(A)); sys.exit(4)
new = data.replace(A, B)
assert len(new) == len(data)
if not os.path.exists(lib + ".orig"):
    shutil.copy2(lib, lib + ".orig")
tmp = lib + ".tmp"
with open(tmp, "wb") as f:
    f.write(new); f.flush(); os.fsync(f.fileno())
os.chmod(tmp, os.stat(lib).st_mode & 0o7777)
os.replace(tmp, lib)
print("patched", hashlib.sha256(new).hexdigest()); sys.exit(0)
