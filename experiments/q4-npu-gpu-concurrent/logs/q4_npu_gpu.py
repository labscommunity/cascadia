"""Q4: microbenchmark NPU + GPU concurrent inference on charlie.

Per the agent's research, OV 2024.4+ added Lunar Lake NPU/GPU memory sharing
via Level Zero RemoteTensor (zero-copy host-shared). The python autolab d4
finding of -75% with NPU draft + GPU target was OV 2024.x.

This test:
  1. Compile Llama 3.2 1B INT4 (small enough for NPU) on charlie NPU
  2. Compile Llama 3.1 8B INT4 on charlie GPU
  3. Run them CONCURRENTLY (separate threads) and measure aggregate tok/s
  4. Compare to running each device alone

If OV 2026.1 has fixed cross-device sync, concurrent should be near sum-of-devices.
If not, concurrent will be near slower-of-devices (cross-device contention).
"""
import time
import threading
import numpy as np
import openvino as ov

# These are charlie's models — we need the full IRs to run as ov-genai-style monolithic
# For NPU, we need static-shape models. The 1B Llama Intel IR is typically static-shape.
LLAMA_1B_XML = r"C:\cascadia\models\srang992-llama-3.2-1b-int4-ov\openvino_model.xml"
LLAMA_8B_XML = r"C:\cascadia\models\llama-3.1-8b-int4\openvino_model.xml"

PROMPT = np.array([[128000, 70869, 279, 1401, 12062, 1990, 36821]], dtype=np.int64)
N_DECODE = 32

def benchmark_decode(req, prompt, n_decode):
    """Run prefill + n_decode steps, return ms/token."""
    n0 = prompt.shape[1]
    beam = np.zeros(1, dtype=np.int32)
    req.reset_state()
    req.set_input_tensor(0, ov.Tensor(prompt))
    req.set_input_tensor(1, ov.Tensor(np.ones((1, n0), dtype=np.int64)))
    req.set_input_tensor(2, ov.Tensor(np.arange(n0, dtype=np.int64).reshape(1, n0)))
    req.set_input_tensor(3, ov.Tensor(beam))
    req.infer()

    t0 = time.perf_counter()
    for i in range(n_decode):
        req.set_input_tensor(0, ov.Tensor(np.array([[100]], dtype=np.int64)))
        req.set_input_tensor(1, ov.Tensor(np.ones((1, n0+1+i), dtype=np.int64)))
        req.set_input_tensor(2, ov.Tensor(np.array([[n0+i]], dtype=np.int64)))
        req.set_input_tensor(3, ov.Tensor(beam))
        req.infer()
    dt = time.perf_counter() - t0
    return dt / n_decode * 1000

def main():
    core = ov.Core()
    print(f"Available devices: {core.available_devices}", flush=True)
    if "NPU" not in core.available_devices:
        print("NPU not available — skipping", flush=True)
        return

    print(f"\n=== Step 1: GPU alone (Llama 8B) ===", flush=True)
    try:
        gpu_8b = core.compile_model(LLAMA_8B_XML, "GPU")
        gpu_req = gpu_8b.create_infer_request()
        ms_per_tok_gpu = benchmark_decode(gpu_req, PROMPT, N_DECODE)
        print(f"  GPU 8B: {ms_per_tok_gpu:.2f} ms/token = {1000/ms_per_tok_gpu:.2f} tok/s", flush=True)
    except Exception as e:
        print(f"  GPU 8B FAILED: {e}", flush=True)
        ms_per_tok_gpu = None

    print(f"\n=== Step 2: NPU alone (Llama 1B) — static shape compile ===", flush=True)
    # NPU requires static shapes — reshape model to fixed prompt + decode size before compile.
    # We'll bucket by total seq len; for this benchmark, fix at prompt_len + N_DECODE.
    fixed_total = PROMPT.shape[1] + N_DECODE
    try:
        model_1b = core.read_model(LLAMA_1B_XML)
        # Inspect inputs to know shapes
        for p in model_1b.inputs:
            print(f"  input: {sorted(p.get_names())} shape={p.get_partial_shape()}", flush=True)
        # Reshape: input_ids/attention_mask/position_ids will be [1, total_seq]; beam_idx [1]
        # For decode loop we'd need [1, 1] — but for prefill we need [1, fixed_total].
        # Simplest: reshape for PREFILL only and time prefill, since per-token decode dynamic-shape recompile would dominate.
        model_1b.reshape({
            "input_ids": [1, fixed_total],
            "attention_mask": [1, fixed_total],
            "position_ids": [1, fixed_total],
            "beam_idx": [1],
        })
        print(f"  reshaped to fixed total seq = {fixed_total}", flush=True)
        npu_1b = core.compile_model(model_1b, "NPU")
        print(f"  NPU compile OK", flush=True)
        npu_req = npu_1b.create_infer_request()
        # Run a single prefill to estimate per-prefill cost
        beam = np.zeros(1, dtype=np.int32)
        prefill_in = np.tile(PROMPT, (1, fixed_total // PROMPT.shape[1] + 1))[:, :fixed_total]
        attn = np.ones((1, fixed_total), dtype=np.int64)
        pos = np.arange(fixed_total, dtype=np.int64).reshape(1, -1)
        npu_req.set_input_tensor(0, ov.Tensor(prefill_in.astype(np.int64)))
        npu_req.set_input_tensor(1, ov.Tensor(attn))
        npu_req.set_input_tensor(2, ov.Tensor(pos))
        npu_req.set_input_tensor(3, ov.Tensor(beam))
        # Warmup
        npu_req.infer()
        # Time
        t0 = time.perf_counter()
        for _ in range(5):
            npu_req.infer()
        dt = (time.perf_counter() - t0) / 5
        print(f"  NPU 1B single-shot prefill ({fixed_total} tok): {dt*1000:.1f} ms ({fixed_total/dt:.1f} tok/s aggregate)", flush=True)
        ms_per_tok_npu = dt * 1000 / fixed_total  # amortize over prefill
    except Exception as e:
        print(f"  NPU 1B FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        ms_per_tok_npu = None

    if ms_per_tok_gpu is None or ms_per_tok_npu is None:
        print("\nCan't run concurrent test (one device failed alone)", flush=True)
        return

    print(f"\n=== Step 3: GPU + NPU CONCURRENT (decode on GPU, prefill-loop on NPU) ===", flush=True)
    # Re-create requests
    gpu_req2 = gpu_8b.create_infer_request()
    npu_req2 = npu_1b.create_infer_request()

    results = {}
    def gpu_thread():
        results["gpu"] = benchmark_decode(gpu_req2, PROMPT, N_DECODE)
    def npu_thread():
        # NPU does N_DECODE prefills (proxy for "draft work")
        beam = np.zeros(1, dtype=np.int32)
        prefill_in = np.tile(PROMPT, (1, fixed_total // PROMPT.shape[1] + 1))[:, :fixed_total]
        attn = np.ones((1, fixed_total), dtype=np.int64)
        pos = np.arange(fixed_total, dtype=np.int64).reshape(1, -1)
        npu_req2.set_input_tensor(0, ov.Tensor(prefill_in.astype(np.int64)))
        npu_req2.set_input_tensor(1, ov.Tensor(attn))
        npu_req2.set_input_tensor(2, ov.Tensor(pos))
        npu_req2.set_input_tensor(3, ov.Tensor(beam))
        t0 = time.perf_counter()
        for _ in range(5):
            npu_req2.infer()
        results["npu"] = (time.perf_counter() - t0) / 5 * 1000 / fixed_total

    t1 = threading.Thread(target=gpu_thread)
    t2 = threading.Thread(target=npu_thread)
    t0 = time.perf_counter()
    t1.start(); t2.start()
    t1.join(); t2.join()
    wall = time.perf_counter() - t0

    print(f"  GPU 8B (concurrent): {results['gpu']:.2f} ms/token = {1000/results['gpu']:.2f} tok/s", flush=True)
    print(f"  NPU 1B (concurrent): {results['npu']:.2f} ms/token = {1000/results['npu']:.2f} tok/s amortized", flush=True)
    print(f"  Wall time: {wall*1000:.0f} ms", flush=True)

    print(f"\n=== ANALYSIS ===", flush=True)
    print(f"  GPU 8B alone:     {ms_per_tok_gpu:.2f} ms/token", flush=True)
    print(f"  GPU 8B concurrent: {results['gpu']:.2f} ms/token  (slowdown: {(results['gpu']/ms_per_tok_gpu - 1)*100:+.1f}%)", flush=True)
    print(f"  NPU 1B alone:     {ms_per_tok_npu:.2f} ms/token", flush=True)
    print(f"  NPU 1B concurrent: {results['npu']:.2f} ms/token  (slowdown: {(results['npu']/ms_per_tok_npu - 1)*100:+.1f}%)", flush=True)
    print(f"  ", flush=True)
    print(f"  If both slowdowns are <10%, NPU+GPU concurrent on this host has minimal contention.", flush=True)
    print(f"  This unblocks: draft on NPU + target on GPU concurrent execution pattern.", flush=True)

if __name__ == "__main__":
    main()
