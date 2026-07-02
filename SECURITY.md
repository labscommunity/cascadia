# Security

## Reporting a vulnerability

Please report security issues privately via
[GitHub security advisories](https://github.com/labscommunity/cascadia/security/advisories/new)
rather than public issues. We'll acknowledge within a few days and keep
you posted on the fix.

## Threat model

Cascadia's network surface is designed for **trusted LAN deployment** —
think a closet or rack of Intel AI PCs on an isolated subnet, not the
public internet. Hardening focuses on robustness against malformed input
(no panics, no unbounded allocation), **not** on authentication or
transport encryption.

**What you get out of the box:**

* HTTP API: request body cap (default 64 KiB), prompt cap (default
  32 KiB), concurrent-request semaphore (default 16) — all
  operator-configurable. Oversized prompt → 413; over-capacity → 503.
  Engine errors map cleanly to 5xx (no panics).
* Engine queue: pending-task cap on the OpenVINO engines (256);
  `EngineError::QueueFull` → 503.
* TCP relay: 256 MiB cap on tensor payloads, 64 KiB cap on raw control
  recvs, shape × dtype overflow check before alloc, 60 s read timeout
  on every recv. A wedged or hostile peer can't pin a worker thread or
  trigger a multi-GB allocation.
* C++ shim: null pointer guards on every exported function, bounded
  property dicts (256 pairs max), uniform `catch (...)` so C++
  exceptions can't unwind into Rust UB, tensor-shape overflow check.
* Numerics: NaN-aware `argmax` (warns instead of silently returning
  token 0 on a broken forward pass). Rotary `compute()` clamps `start`
  to 16 M positions and `seq_len` to 1 M tokens.
* Registry (`cascadia-download`): atomic write (tmp + fsync + rename),
  reject symlink at registry path, 16 MiB cap, parse errors are hard
  failures.

**What you do NOT get:**

* No TLS on either the HTTP API or the inter-stage TCP relay.
* No client authentication on the HTTP API.
* No mDNS authentication.
* No supply-chain pinning beyond `Cargo.lock`.

## Deploying beyond a trusted LAN

Terminate TLS + auth at a reverse proxy in front of the `--api` port,
and firewall the inter-stage TCP ports (`--listen` / `--next`) so only
sibling workers can reach them.
