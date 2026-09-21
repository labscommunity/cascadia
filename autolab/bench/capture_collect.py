#!/usr/bin/env python3
"""Collect API text and completed final-state files, batch by batch, after settling and gating.

Raw outputs stay outside the repository. Convert and match each capture to
its request with research/mtp_offline/fleet_convert.py on the build host.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import lab


def index(box):
    try:
        with lab.OPENER.open(lab.API + '/api/fleet/capture/%d/index.jsonl' % box, timeout=20) as r:
            return [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def raw_completion(prompt, tokens):
    """The original rendered prompt, byte for byte, without a second template."""
    body = json.dumps(dict(model='inkling', prompt=prompt, stream=True,
                           max_tokens=tokens, temperature=0)).encode()
    request = urllib.request.Request(lab.API + '/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
    text, count = [], 0
    with lab.OPENER.open(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            item = json.loads(data)
            if 'error' in item:
                raise RuntimeError(str(item['error']))
            choice = item.get('choices', [{}])[0]
            count += item.get('n_tokens', 0 if choice.get('finish_reason') else 1)
            if choice.get('text'):
                text.append(choice['text'])
    return dict(text=''.join(text), tokens=count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('corpus', help='earlier corpus JSONL with i and prompt')
    ap.add_argument('out', help='directory outside the repository')
    ap.add_argument('--prompts', type=int, default=36)
    ap.add_argument('--streams', type=int, default=12)
    ap.add_argument('--tokens', type=int, default=160)
    ap.add_argument('--box', type=int, default=10)
    ap.add_argument('--rendered', help='original study prompts JSONL with i and rendered text; use /v1/completions')
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.is_relative_to(Path(lab.LAB).parent.resolve()):
        ap.error('raw state captures must stay outside the repository')
    out.mkdir(parents=True, exist_ok=True)
    corpus = [json.loads(line) for line in Path(a.corpus).read_text().splitlines()][:a.prompts]
    rendered = {r['i']: r['text'] for r in (json.loads(l) for l in Path(a.rendered).read_text().splitlines())} if a.rendered else {}
    telemetry = lab.Telemetry(str(out / 'telemetry.jsonl'))
    telemetry.start()
    started = time.monotonic()
    try:
        for lo in range(0, len(corpus), a.streams):
            batch_dir = out / ('batch-%03d' % lo)
            if (batch_dir / 'complete.json').exists():
                lab.log('capture batch already complete:', lo)
                continue
            batch_dir.mkdir(exist_ok=True)
            before = {r['file'] for r in index(a.box)}
            batch = corpus[lo:lo + a.streams]

            def one(item):
                result = raw_completion(rendered[item['i']], a.tokens) if rendered else lab.chat(item['i'], a.tokens, timeout=900, prompt=item['prompt'])
                return dict(i=item['i'], family=item['i'] % 12, prompt=item['prompt'], response=result)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as ex:
                responses = list(ex.map(one, batch))
            (batch_dir / 'responses.json').write_text(json.dumps(responses, indent=1))
            if any('error' in r['response'] for r in responses):
                raise RuntimeError('API request failed; responses saved, stopping collection')
            deadline = time.monotonic() + 60
            while True:
                new = [r for r in index(a.box) if r['file'] not in before]
                if len(new) >= len(batch):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('missing completed captures; check budget and capture-server state')
                time.sleep(2)
            for record in new:
                name = record['file']
                if Path(name).name != name:
                    raise ValueError('invalid capture filename')
                with lab.OPENER.open(lab.API + '/api/fleet/capture/%d/%s' % (a.box, name), timeout=120) as r:
                    data = r.read(64 * 1024 * 1024)
                if len(data) != record['size']:
                    raise ValueError('capture size mismatch')
                (batch_dir / name).write_bytes(data)
            (batch_dir / 'complete.json').write_text(json.dumps(new, indent=1))
            lab.log('captured batch %d: %d requests, %d files, %.1f MiB' % (
                lo, len(batch), len(new), sum(r['size'] for r in new) / 2**20))
    finally:
        telemetry.stop_ev.set()
        telemetry.join(timeout=25)
        lab.log('capture collection elapsed %.1f seconds' % (time.monotonic() - started))


if __name__ == '__main__':
    main()
