# Deploying tahoma workers

Tahoma is a long-running CLI process; it does not daemonize itself. Run it under whatever supervisor your platform already uses — systemd on Linux, NSSM or Task Scheduler on Windows, launchd on macOS.

## Linux (systemd)

A template unit lives at [`tahoma-worker.service`](tahoma-worker.service). It expects a `tahoma` user, a checkout/install at `/opt/tahoma`, and shards under `/opt/tahoma/shards/`.

```bash
# Adjust the Environment= lines in the unit file, then:
sudo cp docs/deploy/tahoma-worker.service /etc/systemd/system/tahoma-worker@.service
sudo mkdir -p /run/tahoma && sudo chown tahoma:tahoma /run/tahoma
sudo systemctl daemon-reload

# Start the worker for stage 0 and stage 1:
sudo systemctl enable --now tahoma-worker@0.service
sudo systemctl enable --now tahoma-worker@1.service

# Inspect:
sudo systemctl status tahoma-worker@0.service
sudo journalctl -u tahoma-worker@0.service -f
```

The unit is `Type=simple` and uses `--pid-file /run/tahoma/tahoma-worker-%i.pid`. Tahoma writes its PID on start and removes it via `atexit` on clean exit. systemd also restarts the worker on failure (`Restart=on-failure`, capped at 3 starts/min).

`KillSignal=SIGTERM` works with the SIGTERM handler in `cli.cmd_worker` — the worker calls `runner.close()` (drops sockets, releases GPU contexts) before exiting.

## Windows (NSSM)

```powershell
# Install nssm (https://nssm.cc), then:
nssm install tahoma-worker-0 "C:\Python311\python.exe" `
    "-m tahoma worker --rank 0 --total 2 --engine ov-runtime --device GPU " `
    "--model C:\tahoma\shards --next 10.0.0.2:9100 --listen :9100 " `
    "--pid-file C:\ProgramData\tahoma\worker-0.pid --log-level INFO"
nssm set tahoma-worker-0 AppStdout C:\ProgramData\tahoma\worker-0.log
nssm set tahoma-worker-0 AppStderr C:\ProgramData\tahoma\worker-0.log
nssm set tahoma-worker-0 AppExit Default Restart
nssm start tahoma-worker-0
```

NSSM sends `SIGTERM`-equivalent on stop (`Process` graceful shutdown then `WM_CLOSE` then `TerminateProcess`) which our handler picks up.

> Note: Windows OpenSSH spawns processes in Session 0 / Services context that die when the SSH session closes. Always use NSSM (or Task Scheduler `/RU SYSTEM`) for production workers — never rely on `start /B` over SSH.

## macOS (launchd)

For dev only — Tahoma is Intel-Linux-target. A minimal `~/Library/LaunchAgents/com.tahoma.worker.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tahoma.worker</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/tahoma</string>
    <string>worker</string>
    <string>--rank</string><string>0</string>
    <string>--total</string><string>1</string>
    <string>--engine</string><string>ov-optimum</string>
    <string>--model</string><string>unsloth/Meta-Llama-3.1-8B-Instruct</string>
    <string>--api</string><string>:8000</string>
    <string>--pid-file</string><string>/tmp/tahoma-worker.pid</string>
  </array>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/tahoma-worker.out</string>
  <key>StandardErrorPath</key><string>/tmp/tahoma-worker.err</string>
</dict>
</plist>
```

Load with `launchctl load ~/Library/LaunchAgents/com.tahoma.worker.plist`.

## Health checks

The HTTP API exposes:

- `GET /health` → `{"status": "ok"}`
- `GET /v1/models` → list of served model ids

For supervisors that probe via TCP or HTTP, point them at the API port. For pipeline workers (rank > 0) without an API, supervisors should rely on PID file presence + the process exit code.

## Logs

Tahoma logs to stdout/stderr in plain text:

```
2026-05-01 20:41:23,189 INFO tahoma.worker.runner | runner ready
```

For structured log shipping (Loki, CloudWatch), wrap the `tahoma worker` command with whatever envelope your shipper expects — Tahoma intentionally does not bundle a JSON formatter.
