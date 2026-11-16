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

* HTTP API: request body cap (64 KiB), prompt cap (32 KiB),
  concurrent-request semaphore (16). These are library-config defaults
  (`cascadia-api` `Config`); the shipped CLI does not currently expose
  flags to change them. Oversized prompt → 413; over-capacity → 503.
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

## Robot policy endpoints (`ov-policy`)

`cascadia run --engine ov-policy` serves `GET /v1/policy/metadata` and the
WebSocket `/v1/policy/ws`, and with `--zenoh` Intel's physicalai protocol.
Their replies become a robot's motor commands, and they have **no
authentication and no TLS**, like the chat API. So:

* The policy API binds `127.0.0.1:8000` unless `--api` is given, and
  `--zenoh` binds loopback unless `--zenoh-allow-all-interfaces` is given.
  Bound to a network address, anyone who can reach the port can drive the
  robot; OSS single-box mode has no way to restrict who calls.
* A WebSocket upgrade whose `Origin` is not in `--policy-allow-origin`
  (empty by default) is refused, so a browser page cannot open the socket.
  Clients that send no `Origin` are allowed: this is not authentication.
* Inputs are bounded before any work: per-message size cap from the
  policy's spec, 1 KiB prompt, 4 connections (503 beyond), checked array
  sizes before allocation, and a missed ping closes the connection.
* A `policy_package` is code-adjacent: OpenVINO parses and compiles its
  IRs and its manifest decides how actions are scaled. Load packages only
  from sources you trust, and compare the metadata's `package_sha256`
  with the one you vetted.

To serve a robot over a network, keep the server on loopback and put an
authenticated TLS hop in front, or use `cascadia policy-bridge
--token-file` against a gateway that checks the token. The bridge's own
port is loopback and unauthenticated: any process on the bridge's host can
call the remote policy through it, with the bridge's token. Run the bridge
only on a host you trust, such as the robot's own machine. See
[docs/engines/ov-policy.md](docs/engines/ov-policy.md#security).

## Deploying beyond a trusted LAN

Terminate TLS + auth at a reverse proxy in front of the `--api` port,
and firewall the inter-stage TCP ports (`--listen` / `--next`) so only
sibling workers can reach them.
