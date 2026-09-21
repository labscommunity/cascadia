#!/usr/bin/env python3
"""One-box attention projection microbenchmark, before its serving worker loads.

Builds candidate IRs in memory from the box's own shell files. Never installs
them or changes the serving model. Run under a bounded, once-only wrapper.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import numpy as np
import openvino as ov
from inkling_attn_ov import layer_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--layers', default='30,31')
    ap.add_argument('--device', default='GPU')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    core = ov.Core()
    rng = np.random.default_rng(41)
    records = []
    for layer in map(int, a.layers.split(',')):
        reference = {}
        for bits in (8, 4):
            q, o, weights, wo = layer_models(a.model, layer, 'int%d' % bits)
            hidden, context = weights[0][1].shape[1], wo.shape[1]
            x = (rng.standard_normal((1, 2, hidden)) * 0.05).astype(np.float32) if bits == 8 else x
            ctx = (rng.standard_normal((1, 2, context)) * 0.05).astype(np.float32) if bits == 8 else ctx
            compiled_q = core.compile_model(q, a.device, {'INFERENCE_PRECISION_HINT': 'f16'})
            compiled_o = core.compile_model(o, a.device, {'INFERENCE_PRECISION_HINT': 'f16'})
            rq, ro = compiled_q.create_infer_request(), compiled_o.create_infer_request()
            for rows in (1, 2):
                args_q, args_o = {'x': x[:, :rows]}, {'ctx': ctx[:, :rows]}
                for _ in range(5):
                    rq.infer(args_q); ro.infer(args_o)
                qs, os = [], []
                for _ in range(41):
                    start = time.perf_counter_ns(); yq = rq.infer(args_q)
                    qs.append((time.perf_counter_ns() - start) / 1000)
                    start = time.perf_counter_ns(); yo = ro.infer(args_o)
                    os.append((time.perf_counter_ns() - start) / 1000)
                outputs = [np.array(v, copy=True).reshape(-1) for v in list(yq.values()) + list(yo.values())]
                if not all(np.isfinite(v).all() for v in outputs):
                    raise ValueError('non-finite projection output')
                if bits == 8:
                    reference[rows] = outputs
                diff2 = sum(float(np.square(v.astype(np.float64)-b).sum()) for v,b in zip(outputs,reference[rows]))
                ref2 = sum(float(np.square(b.astype(np.float64)).sum()) for b in reference[rows])
                rec = dict(layer=layer, bits=bits, rows=rows, q_us=round(statistics.median(qs)),
                           o_us=round(statistics.median(os)), relative_rms_ppm=round(1e6*np.sqrt(diff2/max(ref2,1e-30))))
                records.append(rec)
                line = 'AQ%dB%dR%d probe stage profile ' % (layer,bits,rows) + ' '.join('%s=%s' % kv for kv in rec.items())
                # Repeat for the journal follower, with the same de-duplicated tag.
                for _ in range(3):
                    print(line, flush=True); time.sleep(1)
            del q, o, weights, wo, compiled_q, compiled_o, rq, ro, yq, yo
            gc.collect()
    Path(a.out).write_text(json.dumps(records, indent=2) + '\n')
    print('AQ-done probe stage profile complete=1 measurements=%d' % len(records), flush=True)


if __name__ == '__main__':
    main()
