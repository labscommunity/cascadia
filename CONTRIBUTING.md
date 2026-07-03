# Contributing to Cascadia

Thanks for hacking on Cascadia. This is the short version of what you
need to be productive. By participating you agree to our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Get a build running

Start with **[QUICKSTART.md](QUICKSTART.md)** — the 5-minute stub build
needs nothing but Rust and gives you a working end-to-end path before you
touch the OpenVINO stack. For real inference, **[INSTALL.md](INSTALL.md)**
covers the OpenVINO SDK + GPU runtime.

```bash
cargo build --release -p cascadia    # stub mode
cascadia doctor                       # sanity-check your environment
```

## Repo layout

Cascadia is a Cargo workspace; one concern per crate. The `Engine` +
`Builder` traits in `cascadia-engine` are the plugin seam — engines must
not depend on each other. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
for the design rationale, crate responsibilities, and the wire format.

## Before you commit

CI runs the stub-mode gate below; run it locally first. If you have
[`just`](https://github.com/casey/just):

```bash
just check        # fmt --check + clippy + test, exactly what CI runs
```

Or by hand:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace --all-targets
```

OpenVINO-linked code paths aren't exercised in CI (no Intel GPU on the
runners) — if you change an OV engine, test it on real hardware and say
so in the PR.

## Commit & PR conventions

These are **non-negotiable** for this repo:

- **[Conventional Commits](https://www.conventionalcommits.org/):**
  `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
- **One logical change per commit.** Smaller is better.
- **Don't skip hooks** (`--no-verify`) and don't bypass commit signing.
- **Branch + PR is the unit of work.** Never push to `main`. Branch off
  `main` (`feat/<short-desc>-<issue#>`), open a PR into `main`, reference
  the issue.

## Filing issues

Search existing issues/PRs first. For OpenVINO-related bugs, note the
exact OpenVINO version you reproduce on (the runtime moves fast and many
issues are version-specific).
