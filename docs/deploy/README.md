# Deploying cascadia workers

Cascadia is a long-running CLI process; it does not daemonize itself. Run it under whatever supervisor your platform already uses — systemd on Linux, NSSM or Task Scheduler on Windows, launchd on macOS.

Deploy a **release bundle**: the OpenVINO libraries sit beside the binary, so a service finds them with no extra environment. A **source-built** binary links against the SDK, and a service does not inherit your shell's `PATH` — point it at the runtime libs — **including TBB, which lives beside the runtime, not inside it**: `nssm set <svc> AppEnvironmentExtra PATH=<sdk>\runtime\bin\intel64\Release;<sdk>\runtime\3rdparty\tbb\bin;…` on Windows, `Environment=LD_LIBRARY_PATH=<sdk>/runtime/lib/intel64:<sdk>/runtime/3rdparty/tbb/lib` on Linux. Otherwise the worker exits immediately on a missing `libtbb`/`tbb12`.

## Linux (systemd)

A template unit lives at [`cascadia-worker.service`](cascadia-worker.service). It expects a `cascadia` user, the `cascadia` binary on `PATH` (or adjust `ExecStart`), and shards under `/opt/cascadia/shards/`.

`CASCADIA_API` is what keeps rank 0 alive: a worker started without `--api` reads stdin, gets EOF under systemd, and exits 0 — which `Restart=on-failure` will *not* restart. Relay ranks (rank > 0) ignore `--api` and stay up on the relay loop, so one template serves every rank.

```bash
# Adjust the Environment= lines in the unit file, then:
sudo cp docs/deploy/cascadia-worker.service /etc/systemd/system/cascadia-worker@.service
sudo systemctl daemon-reload

# Start the worker for stage 0 and stage 1 (one per HOST — two instances on the
# same host would both bind CASCADIA_LISTEN):
sudo systemctl enable --now cascadia-worker@0.service
sudo systemctl enable --now cascadia-worker@1.service

# Inspect:
sudo systemctl status cascadia-worker@0.service
sudo journalctl -u cascadia-worker@0.service -f
```

The unit is `Type=simple`, so systemd tracks the worker process directly — no PID file needed. It restarts the worker on failure (`Restart=on-failure`, capped at 3 starts/min).

`KillSignal=SIGTERM` works with rank 0's SIGTERM handler — it calls `runner.close()` (drops sockets, releases GPU contexts) before exiting 0. Relay ranks (rank > 0) sit in the activation-relay loop and do not install a handler, so they are terminated by the signal without a graceful close; nothing is persisted there, but expect no shutdown log line from them.

## Windows (NSSM)

```powershell
# Install nssm (https://nssm.cc), then:
nssm install cascadia-worker-0 "C:\cascadia\cascadia.exe" `
    "worker --rank 0 --total 2 --engine ov-runtime --device GPU " `
    "--model C:\cascadia\shards --next 10.0.0.2:9100 --listen :9100 " `
    "--api :8000 --log-level info"
nssm set cascadia-worker-0 AppStdout C:\ProgramData\cascadia\worker-0.log
nssm set cascadia-worker-0 AppStderr C:\ProgramData\cascadia\worker-0.log
nssm set cascadia-worker-0 AppExit Default Restart
nssm start cascadia-worker-0
```

`--api` is required on rank 0 for the same reason as under systemd: without it the worker reads stdin, gets EOF as a service, and exits 0 — which `AppExit Default Restart` turns into a restart loop. Relay ranks (rank > 0) don't need it.

NSSM's graceful-stop sequence (console event, then `WM_CLOSE`, then `TerminateProcess`) gives the worker a chance to shut down cleanly.

> Note: Windows OpenSSH spawns processes in Session 0 / Services context that die when the SSH session closes. Always use NSSM (or Task Scheduler `/RU SYSTEM`) for production workers — never rely on `start /B` over SSH.

## macOS (launchd)

For dev only — there is no Intel GPU runtime on macOS, so use the `mock` engine (or a stub build). A minimal `~/Library/LaunchAgents/com.cascadia.worker.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.cascadia.worker</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/cascadia</string>
    <string>worker</string>
    <string>--rank</string><string>0</string>
    <string>--total</string><string>1</string>
    <string>--engine</string><string>mock</string>
    <string>--model</string><string>mock-model</string>
    <string>--api</string><string>:8000</string>
  </array>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/cascadia-worker.out</string>
  <key>StandardErrorPath</key><string>/tmp/cascadia-worker.err</string>
</dict>
</plist>
```

Load with `launchctl load ~/Library/LaunchAgents/com.cascadia.worker.plist`.

## Health checks

The HTTP API exposes:

- `GET /health` → `{"status": "ok"}`
- `GET /v1/models` → list of served model ids

For supervisors that probe via TCP or HTTP, point them at the API port. Pipeline workers (rank > 0) don't serve an API; supervise them by process state and exit code — under systemd `Type=simple` that's automatic, and a TCP probe of the `--listen` port works as a liveness check.

## Logs

Cascadia logs to stdout/stderr in plain text via `tracing` (level set by `--log-level`, default `info`):

```text
2026-07-02T17:41:23.189Z  INFO cascadia_runner: runner ready
```

For structured log shipping (Loki, CloudWatch), wrap the `cascadia worker` command with whatever envelope your shipper expects — Cascadia intentionally does not bundle a JSON formatter.
