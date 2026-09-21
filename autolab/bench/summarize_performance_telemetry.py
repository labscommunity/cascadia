#!/usr/bin/env python3
"""Export numeric role diagnostics for completed survey decode windows only.

Raw telemetry never leaves the private operator directory. Profiles must fit
entirely inside a cohort's common decode window and contain no admissions.
This intentionally excludes startup and drain; counters are not whole-phase
error totals. Correlation with throughput does not establish its cause.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics

from telemetry_analysis import normalize_records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment',default='046_final_performance',nargs='?')
    args=ap.parse_args()
    dest=Path(__file__).resolve().parents[1]/'experiments'/args.experiment
    raw=Path.home()/'inkling-release/autolab-telemetry'/args.experiment/'telemetry.jsonl'
    records=normalize_records([json.loads(line) for line in raw.read_text().splitlines() if line])
    phases=json.loads((dest/'measurements.json').read_text())
    output=[]
    for phase in phases:
        if phase['phase'].startswith('pilot'):continue
        windows=[(c['overlap_start'],c['overlap_end']) for c in phase['cohorts']]
        for role in range(11):
            rr=[r for r in records if r.get('rank')==role]
            profiles=[p for r in rr for p in r.get('profs',[])
                      if p.get('opens',0)==0 and any(
                          lo<=p['lt']-p['window_ms']/1000 and p['lt']<=hi for lo,hi in windows)]
            sysrows=[r['sys'] for r in rr if r.get('sys') and any(lo<=r['lt']<=hi for lo,hi in windows)]
            totals=collections.Counter()
            for p in profiles:
                for k in ['window_ms','frames','rows','compute_ms','ov_attn_ms','ov_moe_ms',
                          'ov_moe_fallbacks','ov_moe_nonfinite','spec_sent','spec_hits','spec_misses']:
                    totals[k]+=p.get(k,0)
            def ratio(key,denom):
                return totals[key]/totals[denom] if totals[denom] else None
            row=dict(phase=phase['phase'],streams=phase['streams'],family=phase['family'],role=role,
                     installed_box=rr[-1].get('box',role) if rr else role,profile_windows=len(profiles),
                     profiled_seconds=totals['window_ms']/1000,frames=totals['frames'],rows=totals['rows'],
                     rows_per_frame=ratio('rows','frames'),compute_ms_per_row=ratio('compute_ms','rows'),
                     attention_ms_per_row=ratio('ov_attn_ms','rows'),moe_ms_per_row=ratio('ov_moe_ms','rows'),
                     moe_fallbacks=totals['ov_moe_fallbacks'],moe_nonfinite=totals['ov_moe_nonfinite'],
                     spec_sent=totals['spec_sent'],spec_hits=totals['spec_hits'],spec_misses=totals['spec_misses'],
                     system_samples=len(sysrows))
            for key in ['pkg_w','mhz','temp','swapin_s']:
                values=[r[key] for r in sysrows if isinstance(r.get(key),(int,float))]
                row['mean_'+key]=statistics.mean(values) if values else None
            available=[r['mem_avail'] for r in sysrows if isinstance(r.get('mem_avail'),(int,float))]
            row['min_mem_available_mib']=min(available) if available else None
            output.append(row)
    with (dest/'role-diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(output)
    print(f'Exported {len(output)} numeric phase/role rows; no raw telemetry fields')


if __name__=='__main__':main()
