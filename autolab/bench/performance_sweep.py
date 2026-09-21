#!/usr/bin/env python3
"""Resumable, token-accounted fleet concurrency and prompt-family measurements.

Run with the operator's measurement venv (aiohttp). No deployment is performed.
Raw responses/events and telemetry stay outside the repository. Completed phase
summaries are written atomically after every phase; any failure stops the suite.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import time

import aiohttp
import collect
import lab

FAMILIES = ['explanation', 'code', 'arithmetic', 'story', 'tips', 'table',
            'rewrite', 'facts', 'poem', 'instructions', 'translation', 'true_false']
LEVELS = [1, 2, 4, 6, 8, 11, 15, 22, 32, 48, 64, 96, 128, 176, 256]
EXPECTED_RELEASE = 1790016660
EXPECTED_BINARY = '9084392040688eaa6aa9ff6cf6d222f84e24e119e528d91920d12ddd106563c2'
EXPECTED_RESTARTS = [2, 5, 5, 4, 3, 3, 3, 2, 0, 1, 1]
CAPTURE_BUDGET_PHASE = ('state capture stopped, incomplete .part files retained: '
                        'state capture byte budget exhausted')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def prompt_for(family, index):
    # Identical prompts across concurrency points and repeat passes. These
    # are first/repeat study exposures, not a claim of never-seen fleet text.
    key = f'046-performance-final/{family}/{index}'
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return collect.FAMILIES[family](h >> 4)


def check_fleet():
    st = lab.status()
    rows = lab.fleet_rows(st)
    if abs(time.time() - st.get('time', 0)) > 120:
        raise RuntimeError('Signed fleet report is stale or the entry clock changed')
    if st.get('poller', {}).get('applied') != EXPECTED_RELEASE:
        raise RuntimeError('Fleet release changed during the study')
    if st.get('fleet', {}).get('files', {}).get('cascadia') != EXPECTED_BINARY:
        raise RuntimeError('Fleet binary hash changed')
    if len(rows) != 11 or any(r['state'] != 'active' for r in rows.values()):
        raise RuntimeError('A fleet worker is no longer active')
    if [rows[k]['restarts'] for k in range(11)] != EXPECTED_RESTARTS:
        raise RuntimeError('A worker restarted since the verified settled deployment')
    if len({r['files'] for r in rows.values()}) != 1:
        raise RuntimeError('Fleet file versions differ')
    # Profiling can fill the beacon journal window and leave an empty phase.
    # Accept that only with the exact release/restart anchor that passed
    # steady 3/3 and both correctness gates on 2026-09-21 at 14:00 CDT.
    # This exact bounded-diagnostic warning sets capture.failed and makes
    # later capture calls no-ops. It neither exits nor changes inference.
    bad = [r['phase'] for r in rows.values()
           if r['phase'] and r['phase'] != CAPTURE_BUDGET_PHASE
           and (not lab.worker_is_serving(r) or 'last error:' in r['phase'])]
    if bad:
        raise RuntimeError('Fleet phase changed: ' + repr(bad))
    return st['time']


def summarize_cohort(rows, start, end):
    lo = max(r['events'][0][0] for r in rows)
    hi = min(r['events'][-1][0] for r in rows)
    duration = max(0, hi - lo)
    window_tokens = sum(n for r in rows for t, n in r['events'] if lo < t <= hi)
    rates = []
    for r in rows:
        events = r['events']
        elapsed = events[-1][0] - events[0][0]
        r['decode_tok_s'] = ((r['tokens'] - events[0][1]) / elapsed if elapsed > 0 else None)
        rates.append(r['decode_tok_s'])
    return dict(start=start, end=end, wall_s=end-start, tokens=sum(r['tokens'] for r in rows),
                overlap_start=lo, overlap_end=hi, overlap_s=duration,
                overlap_tokens=window_tokens,
                steady_aggregate_tok_s=window_tokens/duration if duration else None,
                sum_stream_tok_s=sum(r for r in rates if r is not None))


class Sweep:
    def __init__(self, exp, cap=1500):
        self.exp, self.cap = exp, cap
        self.dest = Path(lab.LAB) / 'experiments' / exp
        self.raw = Path.home() / 'inkling-release/autolab-telemetry' / exp
        self.raw.mkdir(parents=True, exist_ok=True)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.result_path = self.dest / 'measurements.json'
        self.results = json.loads(self.result_path.read_text()) if self.result_path.exists() else []
        self.pending = set()
        self.failure = None
        self.phase_active = False
        self.session = None

    async def stats(self):
        async with self.session.get(lab.API + '/api/stats', timeout=aiohttp.ClientTimeout(total=8)) as r:
            r.raise_for_status()
            return await r.json()

    async def watch(self):
        last_health = 0
        last_report, last_advance = None, time.monotonic()
        try:
            with (self.raw / 'api-stats.jsonl').open('a') as f:
                while True:
                    stats = await self.stats()
                    f.write(json.dumps(dict(time=time.time(), **stats)) + '\n'); f.flush()
                    now = time.monotonic()
                    if now - last_health >= 25:
                        report = await asyncio.to_thread(check_fleet)
                        if report != last_report:
                            last_report, last_advance = report, now
                        elif now - last_advance > 90:
                            raise RuntimeError('Signed report stopped advancing')
                        last_health = now
                    await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.failure = f'{type(e).__name__}: {e}'
            lab.log('STOP health monitor:', self.failure)
            for task in list(self.pending):
                task.cancel()

    async def request(self, family, index, tokens, stagger):
        await asyncio.sleep(stagger)
        if self.failure:
            raise RuntimeError(self.failure)
        started = time.time()
        row = dict(family=FAMILIES[family], family_index=family, prompt_index=index,
                   prompt=prompt_for(family, index), started=started, events=[], text='', usage=None,
                   capacity_retries=0, engine_queue_rejections=0, permit_rejections=0,
                   capacity_retry_sleep_s=0)
        body = dict(model='model', messages=[dict(role='user', content=row['prompt'])],
                    temperature=0, max_tokens=tokens, stream=True, stream_options=dict(include_usage=True))
        seen_done = False
        try:
            while True:
                if self.failure:
                    raise RuntimeError(self.failure)
                remaining=self.cap-(time.time()-started)
                if remaining<=0:
                    raise TimeoutError('Request deadline including capacity queueing exceeded')
                response=await self.session.post(lab.API+'/v1/chat/completions',json=body,
                                                 timeout=aiohttp.ClientTimeout(total=remaining))
                if response.status<400:
                    break
                detail=await response.text()
                response.release()
                try: reason=json.loads(detail).get('error','')
                except (ValueError,AttributeError): reason=''
                queue_full=isinstance(reason,str) and re.fullmatch(r'queue full \(\d+ pending, cap \d+\)',reason)
                permit_full=reason=='engine at capacity; retry after current requests complete'
                if response.status!=503 or not (queue_full or permit_full):
                    raise RuntimeError(f'HTTP {response.status}: {detail[:400]}')
                row['capacity_retries']+=1
                row['engine_queue_rejections' if queue_full else 'permit_rejections']+=1
                row['last_capacity_response']=reason
                if not getattr(self,'capacity_noted',False):
                    self.capacity_noted=True
                    lab.log('Capacity backoff:',reason)
                delay=min(5,0.5+0.25*row['capacity_retries'])*(0.9+(index%5)*0.05)
                row['capacity_retry_sleep_s']+=delay
                await asyncio.sleep(delay)
            async with response:
                row['headers_s']=time.time()-started
                async for line in response.content:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        seen_done = True
                        break
                    v = json.loads(data)
                    if 'error' in v:
                        raise RuntimeError('SSE error: ' + str(v['error']))
                    if v.get('usage'):
                        row['usage'] = v['usage']
                    for choice in v.get('choices', []):
                        delta = choice.get('delta') or {}
                        content = delta.get('content') or delta.get('reasoning_content') or ''
                        # Empty final marker contributes zero; n_tokens is
                        # otherwise authoritative, including speculative batches.
                        n = int(v.get('n_tokens', 1)) if content or choice.get('finish_reason') is None else 0
                        if n > 0:
                            row['events'].append([time.time(), n])
                        row['text'] += content
                        if choice.get('finish_reason') is not None:
                            row['finish_reason'] = choice['finish_reason']
            row['ended'] = time.time()
            row['tokens'] = sum(n for _, n in row['events'])
            if not seen_done or not row['usage'] or not row['events']:
                raise RuntimeError('Incomplete stream, missing usage, or no tokens')
            if row['tokens'] != row['usage']['completion_tokens']:
                raise RuntimeError(f"Token accounting mismatch: events={row['tokens']} usage={row['usage']}")
            row['ttft_s'] = row['events'][0][0] - started
            row['wall_s'] = row['ended'] - started
            row['prompt_tokens'] = row['usage']['prompt_tokens']
            row['chunks'] = len(row['events'])
            return row
        except BaseException as e:
            row['error'] = f'{type(e).__name__}: {e}'
            with (self.raw / 'failed-requests.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            raise

    async def phase(self, name, n, tokens=128, family=None, samples=None):
        if any(r['phase'] == name for r in self.results):
            return next(r for r in self.results if r['phase'] == name)
        if self.failure:
            raise RuntimeError(self.failure)
        await asyncio.to_thread(check_fleet)
        before = await self.stats()
        if before['requests_in_flight']:
            raise RuntimeError('Competing requests detected before phase; pause dashboard Play')
        count = samples or max(n, 12 if family is None else n)
        count = math.ceil(count / n) * n
        lab.log(f'START {name}: {n} streams, {count} requests, {tokens} tokens/request')
        raw_path = self.raw / (name + '.jsonl')
        if raw_path.exists():
            raw_path.rename(self.raw / (name + f'.previous-{time.time_ns()}.jsonl'))
        started = time.time(); rows = []; cohorts = []
        self.phase_active = True
        self.capacity_noted = False
        try:
            for low in range(0, count, n):
                cohort_start = time.time()
                tasks = [asyncio.create_task(self.request(
                    family if family is not None else (j % 12),
                    j if family is not None else j // 12,
                    tokens, (j-low)*0.015)) for j in range(low, low+n)]
                self.pending.update(tasks)
                try:
                    current = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks: task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                finally:
                    self.pending.difference_update(tasks)
                cohorts.append(summarize_cohort(current, cohort_start, time.time()))
                rows.extend(current)
                with raw_path.open('a') as f:
                    for row in current: f.write(json.dumps(row)+'\n')
            end = time.time()
            after = await self.stats()
            total = sum(r['tokens'] for r in rows)
            if after['tokens_total'] - before['tokens_total'] != total:
                raise RuntimeError('Server token counter disagrees: competing traffic or lost token accounting')
            queue_rejections=sum(r['engine_queue_rejections'] for r in rows)
            if after['requests_total'] - before['requests_total'] != count+queue_rejections:
                raise RuntimeError('Server request counter disagrees: competing traffic')
            overlap = sum(c['overlap_s'] for c in cohorts)
            overlap_tokens = sum(c['overlap_tokens'] for c in cohorts)
            rates = [r['decode_tok_s'] for r in rows if r['decode_tok_s'] is not None]
            result = dict(phase=name, family=FAMILIES[family] if family is not None else 'mixed',
                          streams=n, tokens_req=tokens, requests=count, tokens=total,
                          start=started, end=end, wall_s=end-started, aggregate_tok_s=total/(end-started),
                          steady_aggregate_tok_s=overlap_tokens/overlap if overlap else None,
                          steady_per_stream_tok_s=overlap_tokens/overlap/n if overlap else None,
                          overlap_s=overlap, overlap_tokens=overlap_tokens,
                          stream_tok_s_median=quantile(rates,.5), stream_tok_s_p10=quantile(rates,.1),
                          stream_tok_s_p90=quantile(rates,.9), stream_tok_s_max=max(rates),
                          ttft_median_s=quantile([r['ttft_s'] for r in rows],.5),
                          ttft_p95_s=quantile([r['ttft_s'] for r in rows],.95),
                          ttft_max_s=max(r['ttft_s'] for r in rows),
                          prompt_tokens_median=quantile([r['prompt_tokens'] for r in rows],.5),
                          completion_tokens_median=quantile([r['tokens'] for r in rows],.5),
                          early_finish=sum(r['tokens']<tokens for r in rows),
                          capacity_retries=sum(r['capacity_retries'] for r in rows),
                          engine_queue_rejections=queue_rejections,
                          chunks=sum(r['chunks'] for r in rows), cohorts=cohorts,
                          request_metrics=[{k:v for k,v in r.items() if k not in ['text','prompt','events','usage']}
                                           for r in rows])
            self.results.append(result); atomic_json(self.result_path,self.results)
            lab.log(f"DONE {name}: steady {result['steady_aggregate_tok_s']:.2f} total / "
                    f"{result['steady_per_stream_tok_s']:.2f} per stream; end-to-end {result['aggregate_tok_s']:.2f}; "
                    f"TTFT p50/p95 {result['ttft_median_s']:.2f}/{result['ttft_p95_s']:.2f}s; "
                    f"capacity retries {result['capacity_retries']}")
            await asyncio.sleep(5)
            return result
        except BaseException as e:
            atomic_json(self.raw/'interruption.json',dict(phase=name,time=time.time(),error=f'{type(e).__name__}: {e}',
                                                         health_error=self.failure))
            raise
        finally:
            self.phase_active = False

    async def run(self, suite):
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=600),
                                         timeout=timeout, trust_env=False) as self.session:
            await asyncio.to_thread(check_fleet)
            watcher = asyncio.create_task(self.watch())
            telemetry = lab.Telemetry(str(self.raw/'telemetry.jsonl'),every=5)
            telemetry.start()
            try:
                if suite == 'pilot':
                    await self.phase('pilot_single',1,32,family=0)
                    await self.phase('pilot_four',4,32)
                    return
                levels = LEVELS.copy()
                for n in levels:
                    await self.phase(f'mixed_a_c{n:03}',n)
                high = next(r for r in self.results if r['phase']=='mixed_a_c256')
                lower = max(r['steady_aggregate_tok_s'] for r in self.results
                            if r['phase'].startswith('mixed_a') and r['streams']<256)
                if high['steady_aggregate_tok_s'] > lower*1.05:
                    levels.append(352)
                    await self.phase('mixed_a_c352',352)
                for n in reversed(levels):
                    await self.phase(f'mixed_b_c{n:03}',n)
                if suite == 'mixed': return
                means = {n:statistics.mean(r['steady_aggregate_tok_s'] for r in self.results
                         if r['phase'] in [f'mixed_a_c{n:03}',f'mixed_b_c{n:03}']) for n in levels}
                best = max(means,key=means.get)
                family_levels = sorted({1,15,64,best})
                atomic_json(self.dest/'family-plan.json',dict(mixed_peak_streams=best,levels=family_levels))
                for family in range(12):
                    for n in family_levels:
                        await self.phase(f'family_{family:02}_a_c{n:03}',n,family=family,
                                         samples=3 if n==1 else None)
                    await self.phase(f'family_{family:02}_b_c001',1,family=family,samples=3)
                    await self.phase(f'family_{family:02}_b_c{best:03}',best,family=family)
                atomic_json(self.dest/'complete.json',dict(time=time.time(),release=EXPECTED_RELEASE,
                            mixed_levels=levels,family_levels=family_levels,phases=len(self.results)))
            finally:
                watcher.cancel()
                await asyncio.gather(watcher,return_exceptions=True)
                telemetry.stop_ev.set()
                await asyncio.to_thread(telemetry.join,10)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp',default='046_final_performance')
    ap.add_argument('--suite',choices=['pilot','mixed','all'],default='all')
    args=ap.parse_args()
    owner=Path(lab.LOCK).read_text()
    if 'autolab-continuation-20260921' not in owner:
        raise SystemExit('Publisher ownership changed')
    import resource
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(8192,hard),hard))
    asyncio.run(Sweep(args.exp).run(args.suite))


if __name__=='__main__': main()
