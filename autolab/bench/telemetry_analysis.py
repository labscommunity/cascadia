"""Place profiles on the operator's clock and retain installed-box identity.

Worker timestamps can be a day ahead; only the entry box's receive clock
is shared by the profile and its enclosing telemetry sample.
"""
import json


def profile_key(profile):
    at = profile.get('at')
    if at is not None:
        return ('at', str(at))
    return ('body', json.dumps({k: v for k, v in profile.items() if k != 'rt'}, sort_keys=True))


def normalize_records(records):
    roles = {}
    for record in records:
        box = record.get('rank')
        if box is None:
            continue
        if isinstance(record.get('role'), int):
            roles[box] = record['role']
        for profile in record.get('profs', []):
            if profile.get('window_ms', 0) > 0 and isinstance(profile.get('rank'), int):
                roles[box] = profile['rank']
    seen = {}
    result = []
    for record in records:
        box = record.get('rank')
        if box is None:
            result.append(dict(record))
            continue
        out = dict(record, box=box, rank=roles.get(box, box))
        out['profs'] = []
        for profile in record.get('profs', []):
            # Probe tags also contain integers, but are not timing windows.
            if profile.get('window_ms', 0) <= 0:
                continue
            key = profile_key(profile)
            if key in seen.setdefault(box, set()):
                continue
            seen[box].add(key)
            p = dict(profile)
            age = (record['rt'] - p['rt']) if record.get('rt') is not None and p.get('rt') is not None else 0
            p['lt'] = record['lt'] - max(0, age)
            out['profs'].append(p)
        result.append(out)
    return result
