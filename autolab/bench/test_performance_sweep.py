import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web

import performance_sweep as perf


class MetricTests(unittest.TestCase):
    def test_common_window_counts_tokens_not_chunks_and_excludes_first_batch(self):
        rows = [dict(events=[[1,2],[2,3],[4,1]],tokens=6),
                dict(events=[[2,1],[3,2],[5,2]],tokens=5)]
        r = perf.summarize_cohort(rows,0,6)
        self.assertEqual(r['overlap_s'],2)
        self.assertEqual(r['overlap_tokens'],3)
        self.assertEqual(r['steady_aggregate_tok_s'],1.5)
        self.assertAlmostEqual(rows[0]['decode_tok_s'],4/3)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_phase_validates_counters_and_persists_summary(self):
        import json
        import time
        t=time.time()
        def row(index):
            return dict(events=[[t+index*.01,1],[t+1+index*.01,2]],tokens=3,
                        ttft_s=.1,wall_s=1.1,prompt_tokens=12,chunks=2,text='abc')
        with tempfile.TemporaryDirectory() as d, patch.object(perf.lab,'LAB',d), \
             patch.object(perf.Path,'home',return_value=Path(d)), \
             patch.object(perf,'check_fleet',return_value=1), \
             patch.object(perf.asyncio,'sleep',new=AsyncMock()):
            sweep=perf.Sweep('fixture')
            sweep.stats=AsyncMock(side_effect=[dict(requests_in_flight=0,tokens_total=0,requests_total=0),
                                               dict(requests_in_flight=0,tokens_total=6,requests_total=2)])
            sweep.request=AsyncMock(side_effect=[row(0),row(1)])
            result=await sweep.phase('check',2,3,family=0)
            self.assertEqual(result['tokens'],6)
            self.assertGreater(result['wall_s'],0)
            self.assertEqual(json.loads(sweep.result_path.read_text())[0]['phase'],'check')

    async def fetch(self, usage=5, done=True):
        import json
        async def handler(request):
            body=await request.json()
            self.assertTrue(body['stream_options']['include_usage'])
            records=[dict(n_tokens=3,choices=[dict(delta=dict(content='abc'),finish_reason=None)]),
                     dict(n_tokens=2,choices=[dict(delta=dict(content='de'),finish_reason=None)]),
                     dict(n_tokens=1,choices=[dict(delta=dict(content=''),finish_reason='length')]),
                     dict(choices=[],usage=dict(completion_tokens=usage,prompt_tokens=12))]
            data=''.join('data: '+json.dumps(r)+'\n\n' for r in records)
            if done:data+='data: [DONE]\n\n'
            return web.Response(text=data,content_type='text/event-stream')
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as d,patch.object(perf.lab,'API',f'http://127.0.0.1:{port}'):
                sweep=perf.Sweep.__new__(perf.Sweep)
                sweep.failure=None;sweep.cap=5;sweep.raw=Path(d)
                async with aiohttp.ClientSession() as sweep.session:
                    return await sweep.request(0,0,32,0)
        finally:await runner.cleanup()

    async def test_batched_tokens_and_empty_final_marker_match_usage(self):
        row=await self.fetch()
        self.assertEqual(row['tokens'],5)
        self.assertEqual(row['chunks'],2)
        self.assertEqual(row['text'],'abcde')

    async def test_usage_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'accounting mismatch'):
            await self.fetch(usage=6)

    async def test_incomplete_stream_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Incomplete stream'):
            await self.fetch(done=False)


if __name__=='__main__':unittest.main()
