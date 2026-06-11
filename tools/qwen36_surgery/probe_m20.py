# M2'-0 probe items 2+3: per-expert OV call overhead + placement round-trip.
# Synthetic IRs; protocol: 3 warmup + 5 measured runs, median/p10/p90.
import statistics, time
import numpy as np
import openvino as ov
from openvino import opset13 as ops

HIDDEN, INTER, EXPERTS_PER_TOK, LAYERS = 2048, 512, 8, 40
core = ov.Core()
print("devices:", core.available_devices, flush=True)

def expert_model(n_experts=1):
    # x[1,H] -> gate/up [H, I*n] -> mul -> down [I*n, H]
    x = ops.parameter([1, HIDDEN], np.float32, name="x")
    w_gate = ops.constant(np.random.rand(HIDDEN, INTER * n_experts).astype(np.float32) * 0.01)
    w_up = ops.constant(np.random.rand(HIDDEN, INTER * n_experts).astype(np.float32) * 0.01)
    w_down = ops.constant(np.random.rand(INTER * n_experts, HIDDEN).astype(np.float32) * 0.01)
    g = ops.matmul(x, w_gate, False, False)
    u = ops.matmul(x, w_up, False, False)
    h = ops.multiply(ops.sigmoid(g), u)
    y = ops.matmul(h, w_down, False, False)
    return ov.Model([y], [x], f"expert_x{n_experts}")

def shell_model():
    # stand-in for attention/DeltaNet/router compute: 2x [H,H] matmuls
    x = ops.parameter([1, HIDDEN], np.float32, name="x")
    w1 = ops.constant(np.random.rand(HIDDEN, HIDDEN).astype(np.float32) * 0.01)
    w2 = ops.constant(np.random.rand(HIDDEN, HIDDEN).astype(np.float32) * 0.01)
    y = ops.matmul(ops.matmul(x, w1, False, False), w2, False, False)
    return ov.Model([y], [x], "shell")

def bench(fn, warmup=3, runs=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), max(ts)

x_np = np.random.rand(1, HIDDEN).astype(np.float32)

# --- Item 2a: per-expert call overhead (1 expert per OV call, CPU) ---
req1 = core.compile_model(expert_model(1), "CPU").create_infer_request()
CALLS = EXPERTS_PER_TOK * LAYERS  # 320
med, lo, hi = bench(lambda: [req1.infer({"x": x_np}) for _ in range(CALLS)])
print(f"ITEM2a per-token 320 single-expert CPU calls: median {med:.1f} ms (min {lo:.1f} max {hi:.1f}) -> {1000/med:.2f} tok/s ceiling", flush=True)

# --- Item 2b: batched 8-experts-per-call (40 calls/token, CPU) ---
req8 = core.compile_model(expert_model(EXPERTS_PER_TOK), "CPU").create_infer_request()
med, lo, hi = bench(lambda: [req8.infer({"x": x_np}) for _ in range(LAYERS)])
print(f"ITEM2b per-token 40 batched-8 CPU calls:     median {med:.1f} ms (min {lo:.1f} max {hi:.1f}) -> {1000/med:.2f} tok/s ceiling", flush=True)

# --- Item 3: shell->experts->shell placement round-trips, 40 layers ---
for shell_dev in ("CPU", "GPU"):
    try:
        shell_req = core.compile_model(shell_model(), shell_dev).create_infer_request()
    except Exception as e:
        print(f"ITEM3 shell on {shell_dev}: COMPILE FAILED {e}", flush=True)
        continue
    def layer_loop():
        for _ in range(LAYERS):
            shell_req.infer({"x": x_np})           # shell (attn/deltanet/router)
            req8.infer({"x": x_np})                # experts on CPU, batched
    med, lo, hi = bench(layer_loop)
    print(f"ITEM3 {shell_dev}-shell + CPU-experts 40 layers: median {med:.1f} ms (min {lo:.1f} max {hi:.1f}) -> {1000/med:.2f} tok/s ceiling", flush=True)

print("PROBE_DONE", flush=True)
