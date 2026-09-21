#!/usr/bin/env python3
"""Export auditable CSV, PNG/SVG/PDF plots and a report from a performance sweep."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter, NullLocator
import numpy as np

from performance_sweep import FAMILIES, quantile, prompt_for


def save(fig, root, name):
    for ext in ['png','svg','pdf']:
        path=root/f'{name}.{ext}'
        metadata=({'Date':None} if ext=='svg' else
                  {'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
        fig.savefig(path,dpi=180,bbox_inches='tight',metadata=metadata)
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def avg(rows,key):
    return statistics.mean(r[key] for r in rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    args=ap.parse_args(); root=args.directory
    data=json.loads((root/'measurements.json').read_text())
    phases=[r for r in data if not r['phase'].startswith('pilot')]
    if not phases:raise SystemExit('No completed survey phases yet')
    incident_path=root/'capture-budget-event.json'
    capture_event=json.loads(incident_path.read_text()) if incident_path.exists() else None
    for phase in phases:
        phase.setdefault('capacity_retries',0)
        phase.setdefault('engine_queue_rejections',0)
        prompts=[prompt_for(r['family_index'],r['prompt_index']) for r in phase['request_metrics']]
        phase['unique_prompts']=len(set(prompts))
        phase['capture_writes']=('disabled_after_budget' if capture_event and
                                 phase['start']>capture_event['observed_epoch'] else 'enabled')
    fields=['phase','family','streams','requests','tokens_req','tokens','chunks','wall_s',
            'steady_aggregate_tok_s','steady_per_stream_tok_s','aggregate_tok_s','overlap_s',
            'stream_tok_s_median','stream_tok_s_p10','stream_tok_s_p90','stream_tok_s_max',
            'ttft_median_s','ttft_p95_s','ttft_max_s','prompt_tokens_median',
            'completion_tokens_median','early_finish','unique_prompts','capture_writes',
            'capacity_retries','engine_queue_rejections']
    with (root/'phase-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in phases)
    requests=[]
    for phase in phases:
        for r in phase['request_metrics']:
            requests.append(dict(phase=phase['phase'],concurrency=phase['streams'],**r))
    with (root/'request-results.csv').open('w',newline='') as f:
        fields=sorted({k for r in requests for k in r})
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(requests)
    mixed=[r for r in phases if r['family']=='mixed']
    grouped={n:[r for r in mixed if r['streams']==n] for n in sorted({r['streams'] for r in mixed})}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,
                         'svg.hashsalt':'046_final_performance'})
    complete=(root/'complete.json').exists()
    limit_path=root/'concurrency-limit.json'
    limit=json.loads(limit_path.read_text())['max_streams'] if limit_path.exists() else None
    status=('Completed survey' if complete else 'IN PROGRESS — completed phases only')
    paused=(root/'paused.json').exists()
    if paused:status='PAUSED by owner — completed phases only'
    if limit:status+=f' (capped at {limit} streams)'
    report=['# Inkling fleet performance survey','',status,'',
            'Eleven Panther Lake boxes; release `1790016660`; int4 experts and int8 attention. '
            'All survey requests use temperature 0 and a 128-token output budget. '
            'Counts include reasoning and answer tokens.','',
            'Sustained decode is measured only while all requests in each cohort are decoding. '
            'End-to-end throughput includes admission, prefill and drain. Prompts are repeated '
            'across sweep passes; learned phrase history remains enabled. Repeat ranges are '
            'observed variability, not confidence intervals. The fixed prompt pools are finite: '
            'high-concurrency family cohorts can repeat identical prompts. `unique_prompts` in '
            'the phase CSV records diversity; family maxima describe these pools, not a guarantee '
            'for arbitrary novel prompts.','']
    report += ['At high concurrency, the engine’s 64-entry pending queue can reject a burst '
               'before the API’s 512-request limit is reached. Only explicit capacity rejections '
               'are retried with bounded backoff; retry counts are recorded and all queueing '
               'time remains included in TTFT and end-to-end throughput. Other errors stop the run.','']
    if capture_event:
        report += ['The bounded diagnostic capture reached its storage budget during the first '
                   '22-stream attempt. That incomplete attempt was excluded and repeated. Capture '
                   'writes were enabled for the ascending 1–15-stream points and disabled thereafter; '
                   'the reverse sweep and family tests use the same capture-disabled state. '
                   'No worker restarted and no model configuration changed. Repeat differences '
                   'therefore include phrase learning, time/order effects and this instrumentation change.','']
    stress_path=root/'stress-attempts.json'
    if stress_path.exists():
        report += ['Two initial 128-stream attempts were excluded: one received an admission '
                   '503, and the next received an engine no-progress error before generating '
                   'tokens. The workers did not restart. A subsequent correctness gate passed; '
                   'the verified-idle retry then completed all 128 requests without capacity '
                   'retries. These interruptions are retained in [stress attempts](stress-attempts.json). '
                   'Completed-run throughput is not an error-rate or reliability estimate.','']
    if limit:
        report += [f'The 256-stream attempt disconnected during admission. Outstanding requests '
                   f'were cancelled, the unchanged fleet returned to idle, and the correctness '
                   f'gate passed again. The remaining survey was capped at **{limit} streams**. '
                   'The planned 256-stream repeats and conditional 352-stream extension were '
                   'therefore not completed. This is an observed failure, not proof of a hard '
                   'engine concurrency limit; the measured optimum is bounded by the tested range.','']
    if (root/'refinement.json').exists():
        report+=['An **88-stream refinement** was added after the first sweep and the 176-stream '
                 'repeat: it places eight rows in each of eleven pipeline groups. The observed '
                 'slowdown above eight rows per group motivated this extra point. Both 88-stream '
                 'runs occur after the original ascending pass; they are exploratory '
                 'measurements, with the same token and health checks.','']
    if (root/'pause-history.json').exists():
        report+=['The owner paused and later resumed collection. Completed phases were retained '
                 'and the interrupted 88-stream repeat was restarted. The serving release and '
                 'worker restart counts were unchanged. Other generation occurred during the '
                 'pause, so subsequent results also reflect any phrase-history learning from '
                 'that traffic. It is excluded from all measured phase counters. See '
                 '[pause history](pause-history.json).','']
    if grouped:
        ns=list(grouped)
        steady=[avg(grouped[n],'steady_aggregate_tok_s') for n in ns]
        end=[avg(grouped[n],'aggregate_tok_s') for n in ns]
        per=[v/n for n,v in zip(ns,steady)]
        lo=[min(r['steady_aggregate_tok_s'] for r in grouped[n]) for n in ns]
        hi=[max(r['steady_aggregate_tok_s'] for r in grouped[n]) for n in ns]
        best=ns[int(np.argmax(steady))]
        knee=next(n for n,v in zip(ns,steady) if v>=max(steady)*.95)
        best_end=ns[int(np.argmax(end))]
        summary=dict(complete=complete,peak_steady_streams=best,peak_steady_tok_s=max(steady),
                     smallest_streams_within_95pct_peak=knee,peak_end_to_end_streams=best_end,
                     peak_end_to_end_tok_s=max(end),max_streams_limit=limit,paused=paused)
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
        axes[0].plot(ns,steady,'o-',color='#176b78',label='Sustained decode')
        axes[0].fill_between(ns,lo,hi,color='#176b78',alpha=.15,label='Repeat range')
        axes[0].plot(ns,end,'s--',color='#bc6b26',label='Including startup and drain')
        axes[0].axhline(max(steady)*.95,color='#777',ls=':',lw=1,label='95% of observed peak')
        axes[0].set(ylabel='Aggregate output tokens / second',title='Fleet throughput')
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=8)
        axes[1].plot(ns,per,'o-',color='#176b78',label='Sustained total / streams')
        axes[1].fill_between(ns,np.array(lo)/ns,np.array(hi)/ns,color='#176b78',alpha=.15)
        axes[1].set(ylabel='Output tokens / second / stream',title='Per-stream speed as concurrency increases',yscale='log')
        axes[1].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
        axes[1].yaxis.set_minor_locator(NullLocator())
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xlabel('Concurrent streams')
            ticks=[n for n in ns if n in [1,2,4,8,15,32,64,88,128,176,256,352]]
            ax.set_xticks(ticks);ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(axis='x',rotation=35)
        fig.suptitle('Inkling on 11 Panther Lake boxes | '+status,fontsize=13)
        save(fig,root,'concurrency-throughput')
        p50=[];p95=[]
        for n in ns:
            tt=[x['ttft_s'] for r in grouped[n] for x in r['request_metrics']]
            p50.append(quantile(tt,.5));p95.append(quantile(tt,.95))
        fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
        ax.plot(ns,p50,'o-',label='Median');ax.plot(ns,p95,'s-',label='p95')
        ax.set(xscale='log',yscale='log',xlabel='Concurrent streams',ylabel='Seconds to first token',
               title='First-token latency | burst arrival, 15 ms between requests')
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks(ns);ax.xaxis.set_major_formatter(ScalarFormatter());ax.tick_params(axis='x',rotation=45)
        ax.legend();save(fig,root,'concurrency-latency')
        report += [f'Best observed sustained aggregate: **{max(steady):.2f} tok/s at {best} streams**.',
                   f'Smallest tested setting within 95% of that peak: **{knee} streams**.',
                   f'Best observed throughput including startup/drain: **{max(end):.2f} tok/s at {best_end} streams**.','',
                   'These are different objectives from maximizing each user’s speed. '
                   'Use the latency and per-stream columns to choose an operating point.','',
                   '![Concurrency throughput](concurrency-throughput.png)','',
                   '![First-token latency](concurrency-latency.png)','',
                   '| Streams | Runs | Decode tok/s | Decode tok/s/stream | Including startup tok/s | TTFT median / p95 (s) |',
                   '|---:|---:|---:|---:|---:|---:|']
        for i,n in enumerate(ns):
            report.append(f'| {n} | {len(grouped[n])} | {steady[i]:.2f} | {per[i]:.2f} | {end[i]:.2f} | {p50[i]:.2f} / {p95[i]:.2f} |')
        report+=['','| Minimum sustained tok/s/stream | Highest tested concurrency meeting it |',
                    '|---:|---:|']
        for target in [1,2,3,5,10]:
            acceptable=[n for n,v in zip(ns,per) if v>=target]
            report.append(f'| {target} | {max(acceptable) if acceptable else "None"} |')
    families=[r for r in phases if r['family']!='mixed']
    if families:
        names=[f for f in FAMILIES if any(r['family']==f and r['streams']==1 for r in families)]
        medians=[];maxima=[];high_rates=[];high_ns=[]
        for name in names:
            ss=[r for r in families if r['family']==name and r['streams']==1]
            rates=[q['decode_tok_s'] for r in ss for q in r['request_metrics']]
            medians.append(quantile(rates,.5));maxima.append(max(rates))
            choices={n:[r for r in families if r['family']==name and r['streams']==n]
                     for n in {r['streams'] for r in families if r['family']==name}}
            best=max(choices,key=lambda n:avg(choices[n],'steady_aggregate_tok_s'))
            high_ns.append(best);high_rates.append(avg(choices[best],'steady_aggregate_tok_s'))
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
        y=np.arange(len(names))
        axes[0].barh(y,medians,color='#176b78',label='Median single-stream request')
        axes[0].scatter(maxima,y,marker='|',s=150,color='#bc6b26',label='Fastest observed request')
        axes[0].set_yticks(y,names);axes[0].invert_yaxis()
        axes[0].set(xlabel='Output tokens / second',title='Single-stream speed by prompt family');axes[0].legend(fontsize=8)
        axes[1].barh(y,high_rates,color='#176b78');axes[1].set_yticks(y,names);axes[1].invert_yaxis()
        for i,(v,n) in enumerate(zip(high_rates,high_ns)):axes[1].text(v+.3,i,f'{v:.1f} at {n} streams',va='center',fontsize=8)
        axes[1].set_xlim(0,max(high_rates)*1.35)
        axes[1].set(xlabel='Aggregate output tokens / second',title='Best observed family setting (run mean)')
        fig.suptitle(status);save(fig,root,'prompt-families')
        report+=['','![Prompt families](prompt-families.png)','',
                 '| Family | Single-stream median | Fastest single request | Best aggregate decode | Streams at best |',
                 '|---|---:|---:|---:|---:|']
        for f,m,x,h,n in zip(names,medians,maxima,high_rates,high_ns):
            report.append(f'| {f} | {m:.2f} | {x:.2f} | {h:.2f} | {n} |')
        report+=['','Family maxima cover only the tested settings shown in `phase-results.csv`. '
                 'A fastest individual request is sensitive to prompt choice and phrase learning; '
                 'the median and repeated phase means are better deployment expectations.']
    report+=['','Data: [phase CSV](phase-results.csv), [per-request CSV](request-results.csv), '
             '[full sanitized phase measurements](measurements.json). '
             'Each chart is also available as SVG and PDF.','',
             'The exact prompts, generated text, per-event token timing and raw fleet telemetry '
             'are retained privately under the operator’s autolab-telemetry directory. '
             'No host names, addresses or raw telemetry are included here.']
    if (root/'role-diagnostics.csv').exists():
        report+=['','[Role diagnostics](role-diagnostics.csv) contain numeric compute, memory and '
                 'fallback observations. Profile windows must fit entirely inside the shared '
                 'decode interval and contain no admissions. These sampled counters exclude '
                 'startup and drain; their correlations do not establish causes.']
    (root/'report.md').write_text('\n'.join(report)+'\n')
    print(f'Wrote report, CSV and charts for {len(phases)} completed phases to {root}')


if __name__=='__main__':main()
