## Summary

<!-- What does this change and why? Link the issue: "Closes #NN". -->

## Checklist

- [ ] `just check` passes locally (`cargo fmt --all -- --check` + `cargo clippy --workspace --all-targets` + `cargo test --workspace --all-targets`)
- [ ] Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, …), one logical change per commit
- [ ] If this touches an OV engine: tested on real Intel hardware (say which, and which OpenVINO version) — CI only covers stub mode
