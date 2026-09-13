"""Read-only search for existing Inkling checkpoints outside the task directory."""
import json
import os
import time
import sys
from pathlib import Path

found = []
visited = 0
started = time.monotonic()
complete = True
errors = []
prune = {"Windows", "Program Files", "Program Files (x86)", "BuildTools", "msys64",
         "$Recycle.Bin", "System Volume Information", "AppData", "node_modules",
         ".git", "target", "target-msvc", "target-msvc-263", "inkling-autolab",
         "Documents and Settings", "Application Data", "Local Settings", "Intel",
         "DriverBackup", "Recovery", "ProgramData", "npu4621", "opt"}
for directory, dirs, files in os.walk('C:/', onerror=lambda e: errors.append(str(e)), followlinks=False):
    visited += 1
    if time.monotonic() - started > 180:
        complete = False
        break
    allowed = []
    for d in dirs:
        if d in prune:
            continue
        try:
            # Windows junctions are reparse points, but need not be symlinks.
            if getattr(os.lstat(Path(directory, d)), 'st_file_attributes', 0) & 0x400:
                continue
        except OSError:
            continue
        allowed.append(d)
    dirs[:] = allowed
    if visited % 5000 == 0:
        print(f"Scanned {visited} directories", file=sys.stderr, flush=True)
    for name in files:
        if name not in {'manifest.json', 'config.json', 'source_config.json'}:
            continue
        p = Path(directory, name)
        try:
            if p.stat().st_size > 1_000_000:
                continue
            data = json.loads(p.read_text(encoding='utf-8'))
            arch = str(data.get('arch', data.get('model_type', data.get('architectures', '')))).lower()
            if 'inkling' in arch:
                found.append({'path': str(p), 'arch': arch, 'hidden': data.get('hidden_size'),
                              'layers': data.get('num_layers', data.get('num_hidden_layers'))})
        except (OSError, ValueError, AttributeError) as e:
            if 'inkling' in str(p).lower():
                errors.append(str(e))
print(json.dumps({'inkling_manifests': found, 'unreadable_paths': errors, 'directories_scanned': visited, 'completed': complete, 'seconds': time.monotonic() - started}, indent=2))
