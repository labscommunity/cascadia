# Changelog

## [0.2.2](https://github.com/labscommunity/cascadia/compare/v0.2.1...v0.2.2) (2026-08-28)


### Features

* **dist-spec:** concat forced-prefix into draft+target + per-token id surfacing (Option B) ([f78aff0](https://github.com/labscommunity/cascadia/commit/f78aff08779febb8e39a759b4bc0d1dd25de6fe8))
* **engine-mock:** concat+seed resume support (Option B contract vehicle) ([ec5c8ff](https://github.com/labscommunity/cascadia/commit/ec5c8ffb72e3769694add76a4f2c5be09a8509fa))
* **engines:** forced-prefix resume + sparse-moe per-token streaming ([adc15aa](https://github.com/labscommunity/cascadia/commit/adc15aaaec1d0d9d3e8c0b872382a66250a37476))
* **gemma4:** concat forced-prefix + seed generated/last_text (Option B) ([6c0ce8a](https://github.com/labscommunity/cascadia/commit/6c0ce8a110551d00257bca046c1e1f77af1149d3))
* **ov-runtime:** concat forced-prefix + seed generated/last_text (Option B) ([e3b45f6](https://github.com/labscommunity/cascadia/commit/e3b45f6de66729d71a31954782d4a5d6728dcd28))
* **qwen36:** concat forced-prefix + seed; think-splice stays unguarded (Option B) ([b7cb306](https://github.com/labscommunity/cascadia/commit/b7cb3066affc663188a76633e1c4928b1ec3ff24))
* **sparse-moe:** close three observability gaps on the carry paths ([a340fd8](https://github.com/labscommunity/cascadia/commit/a340fd8c136517baa4105f6192a76968e3467a1d))
* **sparse-moe:** concat forced-prefix + seed + greedy-forced resume + per-token id surfacing (Option B) ([2d858a4](https://github.com/labscommunity/cascadia/commit/2d858a4e71778928c8e0027234dc00577a099918))
* **sparse-moe:** log donor-side KV serve refusals + capture stores ([636ddc5](https://github.com/labscommunity/cascadia/commit/636ddc55d11415cf59d630e08a93d98923671a96))
* **sparse-moe:** native cross-chain carried-slice restore (RESTORE_CARRY) ([955e39b](https://github.com/labscommunity/cascadia/commit/955e39bee8d32959205845d4dfb71bf7a3ecaf59))
* **sparse-moe:** prepare_resume helper for streamed Option B seeding ([7cac20a](https://github.com/labscommunity/cascadia/commit/7cac20a209863385ce08325f856277b9894265c2))
* **sparse-moe:** stream OvMoeEngine multi-stage + streamed Option B resume ([2d382c4](https://github.com/labscommunity/cascadia/commit/2d382c4fe3b628c44b580966e851877b9196a207))
* **sparse-moe:** stream SparseMoEEngine multi-stage per-token (Option B) ([f1b6141](https://github.com/labscommunity/cascadia/commit/f1b614182a42db0ade09f974dae29535dba90bcb))
* **tools:** export_kimi_k26 --head — emit head/openvino_model.xml so the native tree serves a LAST stage ([ece6bca](https://github.com/labscommunity/cascadia/commit/ece6bca53a6ea00aca00507f3ddd50c21e009bd1))
* **types:** Chunk.token_ids surfaces every generated id for multi-token chunks (Option B) ([8f124e8](https://github.com/labscommunity/cascadia/commit/8f124e87ad0f9deb8c784e6b1b9e12877289d791))
* **types:** normalize a wire-legal Some([]) resume prefix to None at deserialization ([f62f19b](https://github.com/labscommunity/cascadia/commit/f62f19bd0032f6408f9d6ca2255643d2528fc61e))
* **types:** resume_token_ids + append_resume_ids/resume_generated_seed helpers (Option B) ([7a9e311](https://github.com/labscommunity/cascadia/commit/7a9e311df44c425f32a429ed7faf9275228909dc))


### Bug Fixes

* **dist-spec:** kv_coord capture key excludes seeded resume prefix (Option B+C) ([da8cc6b](https://github.com/labscommunity/cascadia/commit/da8cc6b2e1601e78fe2e2de904167e77f816bb98))
* **dist-spec:** surface round failures instead of finishing cleanly ([4711b11](https://github.com/labscommunity/cascadia/commit/4711b11eecb6e80b93aff1ec28505db7fa59231f))
* **engines:** attribute resume-seed decode failures to the resumed task ([946ae78](https://github.com/labscommunity/cascadia/commit/946ae7894876066c9a93a37c861a1949dbe42758))
* **engines:** final-review wave — pipeline/genai resume declines, streamed seam guard, carry hardening ([9c54d8a](https://github.com/labscommunity/cascadia/commit/9c54d8a85b06164130ddbae21677c282e9c53f8e))
* **engines:** propagate Chunk.token_ids to engine Chunk literals after T2 field add ([5dad4e7](https://github.com/labscommunity/cascadia/commit/5dad4e725b7778e8fddfbd1660d547049e55c783))
* **engines:** resume vocab bound is max assigned id + 1, not the entry count ([ee133a4](https://github.com/labscommunity/cascadia/commit/ee133a468c14579b09df9d4a609a558340c38ff8))
* **engines:** review fixes — attribution, budgets, seams, peer-id validation ([aba7e10](https://github.com/labscommunity/cascadia/commit/aba7e10afc796653cf8d1c04edaf7cba742ab417))
* **engines:** seam divergence must re-anchor, never re-emit the forced prefix (Option B) ([fa327ff](https://github.com/labscommunity/cascadia/commit/fa327ff140e695f7c452a668e6795bdd1937af09))
* **engines:** three review follow-ups on the resume edges ([88d840c](https://github.com/labscommunity/cascadia/commit/88d840c11099b9be8e67163d1313d2c49b0cb83c))
* **gemma4:** resumed prefill folds the seed at T=1 — reproduce the donor, not a new batch shape ([021a14c](https://github.com/labscommunity/cascadia/commit/021a14c2e18eb69c2e2d4901ea803701fe5eedf8))
* **ov-runtime:** packed-slots resume decline carries the resume_unsupported: sentinel ([9e2d291](https://github.com/labscommunity/cascadia/commit/9e2d29187530d0d470d9bf494187aa92ea3df7f9))
* **qwen36:** propagate resume-seed decode error instead of unwrap_or_default (Option B hardening) ([4cb6b84](https://github.com/labscommunity/cascadia/commit/4cb6b849743262e4940aaf2741d3b1efc3373866))
* **sparse-moe:** bound every rank-0 downstream reply wait — a silent peer must error, not stall ([35b498e](https://github.com/labscommunity/cascadia/commit/35b498e23fb59e3d127543ee6ccb7036ec4dfe63))
* **sparse-moe:** cap the decode reply budget at 120s — the activation timeout must not govern it ([9da0316](https://github.com/labscommunity/cascadia/commit/9da0316c5e0f1dee6d93eebe3f0de36f60f76705))
* **sparse-moe:** fix-wave review items R1-R6 (empty prompt, spec-decode resume, warnings, stale docs) ([8375971](https://github.com/labscommunity/cascadia/commit/8375971d5f1b9e96ab93505df3383e8a0d30bda6))
* **sparse-moe:** key multi-stage capture epoch over KV-covered prefix ([053463e](https://github.com/labscommunity/cascadia/commit/053463e78a44e90ba8022ecf89eba8261da381c6))
* **sparse-moe:** log the RestoreCarry relay failure instead of folding it silently ([d583382](https://github.com/labscommunity/cascadia/commit/d583382a1a0c0ad2c90153ff395d1be57eb65fe8))
* **sparse-moe:** mid-stream decode failure surfaces an error, never resets the emit cursor ([c767567](https://github.com/labscommunity/cascadia/commit/c767567418799756d4ea64c33912332b9221e7b2))
* **sparse-moe:** resume budget subtracts prefix len so total==max_tokens (Option B) ([b482951](https://github.com/labscommunity/cascadia/commit/b4829513c891ae072e084f038a8d60bb54258176))
* **sparse-moe:** resume decodes tail with prefix context to preserve the seam (Option B §17) ([00069f1](https://github.com/labscommunity/cascadia/commit/00069f1261210e626b116317b6b77607dd353b3a))
* **sparse-moe:** retire two Option-D follow-ups ([210e7b4](https://github.com/labscommunity/cascadia/commit/210e7b453893b6e1e844e10f8b53df6944006b0f))
* **test:** re-bind the discarded LoadStream in the streaming harnesses ([5486c17](https://github.com/labscommunity/cascadia/commit/5486c17da6e21ce347442ef629554c39ee703da0))
* **test:** un-wedge the gated streaming suites — K26_TINY_HEAD_DIR gate + worker failure surfacing ([9b2048c](https://github.com/labscommunity/cascadia/commit/9b2048c6024e01b9fce42fbedcc4b40689740449))
* **tools:** --head fails fast on missing torch/openvino, and the usage block documents it ([d83a14e](https://github.com/labscommunity/cascadia/commit/d83a14e2596f71270f121b448c923c27595bf4c0))
* **tools:** export K2.6 head IR with bf16 input ([7fae762](https://github.com/labscommunity/cascadia/commit/7fae762c40bc6d8eeec6a7ce83ce23ceb143f183))
* **types:** stamp token_ids on every per-token chunk from resume-capable engines ([2f87f1f](https://github.com/labscommunity/cascadia/commit/2f87f1f24f0303bbc4d05db828726c1d4a7a4a97))
* **types:** supply the base's new tenant field in the Option B test literal ([0d661ba](https://github.com/labscommunity/cascadia/commit/0d661ba4c9c100aa6da7eb4023f5afd33c3df445))


### Refactor

* **sparse-moe:** one capture_multi_stage helper for the two head capture sites ([cb8da09](https://github.com/labscommunity/cascadia/commit/cb8da091b8fa9e0750f44f9b6092f2ff95859730))


### Documentation

* comment-accuracy fixes from the review ([3e7bfa7](https://github.com/labscommunity/cascadia/commit/3e7bfa7a90fc17fe82494f1526935f1fbfa4f693))


### Testing

* **engine-mock:** negative case — shuffled resume seed ids are rejected ([48e068c](https://github.com/labscommunity/cascadia/commit/48e068cf85cc2e50baf25f2d95be7be43c5ef040))
* **engine-mock:** resume seed ids must match the mock's own echo ids ([c5b6de0](https://github.com/labscommunity/cascadia/commit/c5b6de0dbfbc549621dfff6f039b68c64219431a))
* **engines:** close the review-flagged coverage gaps ([c7ec5a6](https://github.com/labscommunity/cascadia/commit/c7ec5a64e44d650c76a059fe125ec0471d92569e))
* **openvino:** serialize activation-timeout knob tests — full-workspace flake ([fbea6c0](https://github.com/labscommunity/cascadia/commit/fbea6c0426e53b8e5326080ff0d1a16394867abb))
* **sparse-moe:** un-gated PipelineEngine resume decline test ([64223a8](https://github.com/labscommunity/cascadia/commit/64223a892058b3d0f1cb3505654e9c6e2f5211f3))


### Miscellaneous

* **engines:** fmt + gate resume_seed_len dead_code under kv_coord (Option B T9 sweep) ([44cbd01](https://github.com/labscommunity/cascadia/commit/44cbd01faa51aa5bcf02827bcdfc09c6ab107d3a))

## [0.2.1](https://github.com/labscommunity/cascadia/compare/v0.2.0...v0.2.1) (2026-08-26)


### Features

* **#34:** cross-chain KV coordination plane ([0438c46](https://github.com/labscommunity/cascadia/commit/0438c46e3154129363720138948c6fa0a52d6b74))
* **cli:** kv_coord feature (multi-stage KV capture/RESTORE + total&gt;1 cache) ([9a9f6f2](https://github.com/labscommunity/cascadia/commit/9a9f6f22a1ef832c344ec7699e477d64685464dc))
* **cli:** kv_coord feature on the cascadia binary crate (propagates to cascadia-cli) ([e071ad9](https://github.com/labscommunity/cascadia/commit/e071ad9ad29747dfac3fe3e893cecb599d515cbf))
* **engine:** gate KvCoordination behind kv_coord feature (keep wire crate out of default trees) ([1c8efc2](https://github.com/labscommunity/cascadia/commit/1c8efc2560cbf9a6365d4418b099710e61a353e8))
* **engine:** KvCoordination surface (issue-34 Option C) — host-side KV export/import ([9df10ab](https://github.com/labscommunity/cascadia/commit/9df10ab690dfce9edcbbf3dbed43df645b0b432f))
* **engine:** KvCoordination::tokenize (head-trigger needs prompt→tokens via the engine's tokenizer) ([0345e96](https://github.com/labscommunity/cascadia/commit/0345e9646f50f1e92ec03c15f86a075dd1e82508))
* **engine:** multi-stage KV capture + head-broadcast epoch (issue-34 Task 1.3, §8) ([bbd6a35](https://github.com/labscommunity/cascadia/commit/bbd6a3520d74f71311b5117411a9784e48a0c7a5))
* **gemma4:** KvCoordination + single-stage warm-pull (issue-34) ([045df22](https://github.com/labscommunity/cascadia/commit/045df225cdc4b0224f9dc58e4428e7d60346a107))
* **gemma4:** multi-stage CAPTURE/RESTORE over frameless transport (issue-34) ([fa21b20](https://github.com/labscommunity/cascadia/commit/fa21b20bb5693f4bc191084edd924a42c8ffe73d))
* **kv-plane:** OvMoe cross-chain multi-stage carried-slice restore ([3a8ac63](https://github.com/labscommunity/cascadia/commit/3a8ac63167e335165aea27fdf25d65165dcbbb13))
* **kv-wire:** append GetV2 with rank (issue-34 1b, donor half) ([b0c4ae3](https://github.com/labscommunity/cascadia/commit/b0c4ae3b3c3700c6fc72fb30cc19d5638a6254ed))
* **kv-wire:** append ReplicatePush/Replicate/ReplicateAck/ReplicaGet/WarmResumeTriggerV2 (issue-34 §12.2) ([475eb63](https://github.com/labscommunity/cascadia/commit/475eb634013230b6baa096bf573643cf6d4d42d4))
* **kv-wire:** appended TenantHint variant (index 16) + exhaustive variant-index tripwire ([ca7fdca](https://github.com/labscommunity/cascadia/commit/ca7fdca9ff03b12c15c8a10183835ad8a3a89a22))
* **kv-wire:** cascadia-kv-wire crate — Manifest/CacheKey/codec/envelope + conformance goldens ([e7e71b7](https://github.com/labscommunity/cascadia/commit/e7e71b720e2f99e1f0c7874f21b076b02e78efea))
* **kv-wire:** KvMessage::Hint(WarmHint) variant (appended; entry→head side-channel, goldens intact) ([d6388bd](https://github.com/labscommunity/cascadia/commit/d6388bd03f298a121cced449c1ce4d13d64df538))
* **kv-wire:** length-delimited KvMessage frame codec (§6/§7 framing, DoS-capped) ([2111dfa](https://github.com/labscommunity/cascadia/commit/2111dfa1bbc0703adf62775f7f0e15b453550c22))
* **kv-wire:** OPAQUE_KV_LAYOUT — relaxed validate branch for opaque snapshots (OV blobs) ([6f0bea7](https://github.com/labscommunity/cascadia/commit/6f0bea71d26f394798b4bfb2a15a25d3b1541ea8))
* **kv:** CaptureV2 carries the head's turn tenant — close the H.1a captures read ([ab33c65](https://github.com/labscommunity/cascadia/commit/ab33c6550b6cf75859bec42ccfd6074b31034074))
* **kv:** emit raw `plane_pulled` at every warm-resume log site ([7c30b19](https://github.com/labscommunity/cascadia/commit/7c30b19042fcb2187f8c0917340cdc4f62ddb259))
* **kv:** give qwen36 + dist-spec worker a plane hand-off mailbox ([564dd72](https://github.com/labscommunity/cascadia/commit/564dd7278e20f000a8cadfa66f3acf00b7a48a86))
* **kv:** H.1 tenant namespacing — sparse-moe engines ([b00a7ef](https://github.com/labscommunity/cascadia/commit/b00a7ef89563d90aa554bfc08efc1fec33884fcf))
* **kv:** H.1a tenant-namespace OvKvCache lookup/serve (openvino engines) ([ed6e67a](https://github.com/labscommunity/cascadia/commit/ed6e67a135ed7b62b1d9f1049e39175a99957093))
* **kv:** H.1b (i) reader half — GenerationTask.tenant + take_warm namespacing ([2cc5c3e](https://github.com/labscommunity/cascadia/commit/2cc5c3e50894b7207fd3d66d9af7b096ebdac64f))
* **kv:** H.1b (ii) partner-bearing CAPTURE body codec (v2) ([4149082](https://github.com/labscommunity/cascadia/commit/4149082c4e4969a5e97e62d61db871251a2de038))
* **kv:** H.1b hard gate — insert keys on the ASSERTED partner, not the manifest echo ([942d80f](https://github.com/labscommunity/cascadia/commit/942d80f77cb3774c6d02358b3d3f49e6bb7581c0))
* **kv:** H.1b R2 — CAPTURE v2 opcode per engine + capture tenant from per-task state ([1b5e4b8](https://github.com/labscommunity/cascadia/commit/1b5e4b84345be5dc28253431a0c42c0a984eee79))
* **kv:** instrument the OvMoe carried-slice path ([fbe8427](https://github.com/labscommunity/cascadia/commit/fbe8427a61d49a497dd42a31c49ff1aeb3220b3b))
* **kv:** kv_holder + apply_warm_resume for gemma4 + qwen36 (Issue-34 plane) ([ea0136a](https://github.com/labscommunity/cascadia/commit/ea0136a64e158817246c9475cd37cfc06ceea6a9))
* **kv:** kv_holder() for ov-dist-spec (driver + worker); model-level fp ([5f19ac6](https://github.com/labscommunity/cascadia/commit/5f19ac64624420336c249de32dac6d6eac6e0e2b))
* **kv:** kv_holder() for sparse-moe (mirror-cache lock-free holder) ([04523fc](https://github.com/labscommunity/cascadia/commit/04523fcf427d72cdb5fe8e37e5532cd28cfbf625))
* **kv:** KV-sharing support — pull only KV-bearing ranks ([140c8ec](https://github.com/labscommunity/cascadia/commit/140c8ec674c0b42c751e20f0e0f23cba01ad48da))
* **kv:** lock-free KV holder handle so a busy engine can serve cross-chain pulls ([69e3526](https://github.com/labscommunity/cascadia/commit/69e35265ab42c89c9cd8326bb9b03f81e9b1cc47))
* **kv:** lock-free warm-handoff mailbox, applied inline before the turn's forward ([24838aa](https://github.com/labscommunity/cascadia/commit/24838aa565de7e1b7c05b02be75227edc3780890))
* **kv:** log NEGOTIATE prefix misses with the divergence point ([b692ef1](https://github.com/labscommunity/cascadia/commit/b692ef17adfed1648977cd1b4ac7ef93950c6bf5))
* **kv:** multi-stage cross-chain warm-resume — distribute pulled per-rank KV ([cd06729](https://github.com/labscommunity/cascadia/commit/cd06729b39e28dbb76e8d98420f454dad59144f2))
* **kv:** plane hand-off mailbox for both sparse-moe engines ([63cbce0](https://github.com/labscommunity/cascadia/commit/63cbce0190b9b43aa1a66610aa3c214e4d29c5aa))
* **kv:** plane warm-resume wire + engine apply_warm_resume (Phase A) ([c5087f5](https://github.com/labscommunity/cascadia/commit/c5087f5760d7d1de9c5b4396178ccf04ed6e4b75))
* **kv:** retract a handed-off warm-resume slice on abort ([f67d1d9](https://github.com/labscommunity/cascadia/commit/f67d1d923ec7b34469eaa8f3b3e6a029103b1673))
* **kv:** sparse-moe multi-stage KV RESTORE (same-chain warm-resume) ([d910903](https://github.com/labscommunity/cascadia/commit/d9109034cdc77ae08844dbc892b3b60be3ab1c50))
* **kv:** step_first plane-restore mode skips chain RESTORE (Phase A.5) ([519b163](https://github.com/labscommunity/cascadia/commit/519b16351cba9cecf1a9bad08e1db1aaf0266565))
* **kv:** time set_state directly at both warm-apply seams ([e27a790](https://github.com/labscommunity/cascadia/commit/e27a790b6e3301740af6c32d7dbc9cf979b7d0db))
* **kv:** track KV-plane provenance on both sparse-MoE prefix caches ([e49e256](https://github.com/labscommunity/cascadia/commit/e49e256dc1e476f9b5afb58c0aad07037e285eb8))
* **kv:** WarmResumeCommit — make the plane warm-resume two-phase ([53c79a4](https://github.com/labscommunity/cascadia/commit/53c79a44597b2a321543d8a8d607cdc9408dd806))
* **kv:** WarmResumeTrigger carries prefix_token_len (asserted downstream GET) — Phase A.7 ([d4fd00f](https://github.com/labscommunity/cascadia/commit/d4fd00f9e2931aa4e1a8a66a0d2c136d4755dd77))
* **ov-dist-spec:** consume warm-resume — multi-stage complete (issue-34) ([b9cce48](https://github.com/labscommunity/cascadia/commit/b9cce48aa175cb405ba04455b5526895349e2b86))
* **ov-dist-spec:** KvCoordination — capture+serve+worker RESTORE handling (issue-34) ([5825e50](https://github.com/labscommunity/cascadia/commit/5825e504441bdfef6ce9fbc177ed6efd8ff99288))
* **ov-kv:** §8 multi-stage CAPTURE foundation — worker stash + transport-agnostic frame codec ([bc8cebf](https://github.com/labscommunity/cascadia/commit/bc8cebffc009534154eaf178382bafe7de1d5e74))
* **ov-runtime:** KvCoordination via opaque KV blobs (issue-34 Option C) ([7fdeb32](https://github.com/labscommunity/cascadia/commit/7fdeb32be43279261bd015548348d4f6709549e5))
* **ov-runtime:** KvCoordination via opaque KV blobs (issue-34 Option C) ([421a3da](https://github.com/labscommunity/cascadia/commit/421a3da5d3e66c1ee3c445325b9d74eaa7dd7229))
* **ov-runtime:** multi-stage §8 CAPTURE over frameless transport (issue-34) ([0a1311d](https://github.com/labscommunity/cascadia/commit/0a1311de892e3f31c049b09234c042dd3be5bb35))
* **ov-runtime:** multi-stage §8 CAPTURE over frameless transport (issue-34) ([72f8a28](https://github.com/labscommunity/cascadia/commit/72f8a2895af5bc42fd9f5feb860bedf009b6a671))
* **ov-runtime:** multi-stage consume RESTORE — warm-resume end-to-end (issue-34) ([5115d5a](https://github.com/labscommunity/cascadia/commit/5115d5a1b460140b9cbb78e16c3e3b285af0279c))
* **ov-shim:** get_state_blob/set_state_blob — opaque KV state export/import for warm-pull (issue-34) ([1c401d3](https://github.com/labscommunity/cascadia/commit/1c401d3dc852656735272d12fb545fd929fad8d9))
* **qwen36:** CASCADIA_QWEN36_RESTORE_RESET — reset_state instead of recreate_request before restore ([5b1a560](https://github.com/labscommunity/cascadia/commit/5b1a560f87817d0b9898fff5eec6211486de9dfd))
* **qwen36:** KvCoordination via multi-stage opaque KV blobs (issue-34) ([f400ecc](https://github.com/labscommunity/cascadia/commit/f400ecccaa0b7400dfed4827af9c0fe3c0165008))
* **qwen36:** opt-in T=1 prefill for byte-identical cross-chain warm-resume ([56a250e](https://github.com/labscommunity/cascadia/commit/56a250ed158bd668e23501bb2ac39c862a7097ac))
* **qwen36:** pipeline §8 CAPTURE broadcast — multi-node chain serves multi-stage KV (issue-34) ([8fe6abc](https://github.com/labscommunity/cascadia/commit/8fe6abc7c40b39fbcaf6c2f3f1727dc642d6eb0c))
* **qwen36:** pipeline consume RESTORE — multi-stage warm-resume end-to-end (issue-34) ([60330ee](https://github.com/labscommunity/cascadia/commit/60330ee00b83554902cf4c0663b461528c2b2697))
* **qwen36:** single-box warm-resume at admission (issue-34) ([2f91fcb](https://github.com/labscommunity/cascadia/commit/2f91fcbc3cf237cbc596299143b82e612c974825))
* **sparse-moe:** multi-stage KV warm-resume on OvMoeEngine (MiniMax-M2) ([96d169c](https://github.com/labscommunity/cascadia/commit/96d169c9f6902a31bf5b189f41daa52cf6f804c7))
* **sparse-moe:** OvMoe cross-chain KV warm-pull (Phase 2a, runtime) ([a406a57](https://github.com/labscommunity/cascadia/commit/a406a57ff1994985b29187e7cf057df086b25731))
* **wire:** WarmResumeTriggerV3 carries each rank's pull-candidate list ([f9e1575](https://github.com/labscommunity/cascadia/commit/f9e15756372821166b2393a949d0d9fc842b2214))


### Bug Fixes

* **ci:** rustfmt, kv-wire license, bincode advisory ([93566e5](https://github.com/labscommunity/cascadia/commit/93566e5cd96acc6d3c71767e4972ced95372e33d))
* **clippy:** never_loop in stub build — recv loops re-iterate only under kv_coord ([3330752](https://github.com/labscommunity/cascadia/commit/3330752a110081e884820f1eb71fdf34492ceb7a))
* **dist-spec:** chunk carried RESTORE blob recv in MAX_RAW_BYTES pieces ([048ab3a](https://github.com/labscommunity/cascadia/commit/048ab3a5432d4d94ca60148ed76999e36f14c85b))
* **dist-spec:** mid-chain RESTORE forwarder was missing carried_len ([e604973](https://github.com/labscommunity/cascadia/commit/e6049739e4b2043c34a2969306a3cdfbaec07e02))
* **dist-spec:** warm-resume must not clamp or guess the draft KV depth ([3ac97ff](https://github.com/labscommunity/cascadia/commit/3ac97ffb3f26efa5650bef3f0ec6a1e9f9e1e1c7))
* **engines:** warm-resume KV-depth off-by-one parity for qwen36/gemma4/dist-spec (issue-34) ([40372b8](https://github.com/labscommunity/cascadia/commit/40372b8b317734898bdff672bd4448e8bd82215e))
* **gemma4:** bound RESTORE/ABORT ack recvs — parity with qwen36's reply_bounded ([8093e02](https://github.com/labscommunity/cascadia/commit/8093e0213764354383a42a7ff93f80622daf0400))
* **gemma4:** recreate request before warm KV restore (mirror qwen36) ([5560b44](https://github.com/labscommunity/cascadia/commit/5560b44ef4afb32db5b43e485545a7f3b52af363))
* **issue34-kv:** compact spec-decode KV at capture so dist-spec warm-resume is safe ([041a541](https://github.com/labscommunity/cascadia/commit/041a541bf4f64c9a6dbe78559ddfb4a02456ff68))
* **kv-plane:** OvMoe multi-stage capture mirrors into holder (cross-chain NEGOTIATE) ([ae698d3](https://github.com/labscommunity/cascadia/commit/ae698d3e36df0921fb2175a70153f1ef14b0fcd0))
* **kv-wire:** bound decode allocation (DoS); gate the KV-state fingerprint diagnostics ([8c7862e](https://github.com/labscommunity/cascadia/commit/8c7862e48f77a143681f4c064b822e1093385a81))
* **kv-wire:** make the alloc-DoS regression test actually exercise the limit ([54a2737](https://github.com/labscommunity/cascadia/commit/54a27373219a644ff0949ab5b7af0dffa167c931))
* **kv:** ABORT rollback scrubs effectively and reports its failures ([f36bb81](https://github.com/labscommunity/cascadia/commit/f36bb817b2c0b0e2efbe92f11f32288eb9194e18))
* **kv:** abort_warm_resume retracts the pulled downstream stashes ([6442ec6](https://github.com/labscommunity/cascadia/commit/6442ec64eab4e9d8aaa3f138531e8cc5961d4e5f))
* **kv:** bound dist-spec's CAPTURE/RESTORE ack waits; drop the connection on timeout ([0addc34](https://github.com/labscommunity/cascadia/commit/0addc34f1123f4dd8cc435818b261cb3018e232a))
* **kv:** bound every RestoreAck wait; drop the connection on timeout ([55d4780](https://github.com/labscommunity/cascadia/commit/55d478060228942c0ea0d3d4a3f59c56ac201b52))
* **kv:** bound ov_blob_decode's layer count before allocating (DoS) ([7ff4b60](https://github.com/labscommunity/cascadia/commit/7ff4b609f500a39b9bf307bda4443e8e5d50fc26))
* **kv:** bump the single-token Forward frame kinds for the push_history byte ([a0e55bd](https://github.com/labscommunity/cascadia/commit/a0e55bdfb2cafadecb5029c1f5e09da61805bb8b))
* **kv:** confine pulled captures to their tenant; log take_warm misses; fix a stale doc comment ([ca4caf0](https://github.com/labscommunity/cascadia/commit/ca4caf0982fcaa4273a704e6fc0aaf43b86b2061))
* **kv:** drop the connection when a CaptureAck wait times out ([5a91f04](https://github.com/labscommunity/cascadia/commit/5a91f04559c9f66d71822e8f11c66e6872eb72aa))
* **kv:** epoch-bind the plane hand-off drain (B1) ([fb478d7](https://github.com/labscommunity/cascadia/commit/fb478d7318f783e5cfebfa002327ef54967c119f))
* **kv:** failed cold-fallback returns Chunk::error, not an empty success ([9f44076](https://github.com/labscommunity/cascadia/commit/9f44076f4fcdfe230f494ca47910434ff87eb4f2))
* **kv:** gate the tracing::error import to kv_coord builds ([37470c8](https://github.com/labscommunity/cascadia/commit/37470c83b31b57b4769f54bc5e673adafce17aef))
* **kv:** key the downstream stash by (epoch, rank), not epoch alone ([6f52537](https://github.com/labscommunity/cascadia/commit/6f5253731d817baa4949881a83f8da600c6488b9))
* **kv:** make a drain-before-first-put visible instead of DEBUG-only ([fba0626](https://github.com/labscommunity/cascadia/commit/fba0626960e77fe8b9cec0566f480a4ccaf0e422))
* **kv:** read CAPTURE bodies in capped chunks; drain over-ceiling blobs before rejecting ([ae19d6a](https://github.com/labscommunity/cascadia/commit/ae19d6a4f558073b9b8b3cb4c62bdf2e9d909820))
* **kv:** refuse an oversized tenant instead of downgrading CaptureV2 to untagged v1 ([332f6fb](https://github.com/labscommunity/cascadia/commit/332f6fb9e176135eeda7bdb70dc24107e5c1f99c))
* **kv:** scrub with recreate_request wherever a set_state_blob is abandoned ([311dde3](https://github.com/labscommunity/cascadia/commit/311dde35dee0b79a657939a77b73db996d40e721))
* **kv:** skip conv/ssm in kv_compact_blob; correct stale qwen36 rationale ([7829470](https://github.com/labscommunity/cascadia/commit/782947074747c107bd3336b59ebf7ef418c73828))
* **kv:** the capture stash must never answer a wire GET ([4288727](https://github.com/labscommunity/cascadia/commit/42887270e7f0fd4f36e62a81368ba63f876a166f))
* **ov-dist-spec:** cross-chain stash_downstream_rank + carry pulled blob to the tail ([c5f2ef0](https://github.com/labscommunity/cascadia/commit/c5f2ef0e421f73af55c50508db8130088af9ef13))
* **ov-runtime:** batched warm-suffix prefill (one forward, not per-token) — issue-34 ([f630b1e](https://github.com/labscommunity/cascadia/commit/f630b1ef43ede28a1cfd67c728c3176dfa85f1fa))
* **ov-runtime:** feed warm-resume suffix token-by-token (rig: chunked prefill unsupported) ([9ce0071](https://github.com/labscommunity/cascadia/commit/9ce0071947bb88a643388a1c6acd86e4dcb4083b))
* **ov-runtime:** ship single stashed downstream blob when RESTORE epoch lookup misses ([033e8d7](https://github.com/labscommunity/cascadia/commit/033e8d778a0b8ddb0d750cd718a25a62348c7858))
* **ov-runtime:** warm-resume position from real KV depth, not token count (issue-34) ([a542c24](https://github.com/labscommunity/cascadia/commit/a542c240ceffab9a8126e527100b399bd97111e2))
* **ov-shim:** set_state_blob restores KV positionally, not by name ([a4f7b82](https://github.com/labscommunity/cascadia/commit/a4f7b826bcbc1f0210ebc19a8e4dfe9045526810))
* **qwen36:** plumb KV_CACHE_PRECISION / DYNAMIC_QUANTIZATION_GROUP_SIZE ([74b7ab9](https://github.com/labscommunity/cascadia/commit/74b7ab92f4cc3dca42e455afd72f502763b397a1))
* **qwen36:** plumb the OV cache settings the CLI never wired; localise bar [#1](https://github.com/labscommunity/cascadia/issues/1) ([e91c2c9](https://github.com/labscommunity/cascadia/commit/e91c2c9e73c946d10dcaa7638683a358d12e9dd7))
* **qwen36:** recreate request before warm KV restore ([b9dfc32](https://github.com/labscommunity/cascadia/commit/b9dfc3290259b83b0e5ac108ffc04d202535b0e4))
* **qwen36:** resume warm KV at true attention depth (ignore fixed-shape SSM states) ([91d1b1b](https://github.com/labscommunity/cascadia/commit/91d1b1bc551d7d450d47862356b8b2c972b08125))
* **rebase:** reconcile the KV plane with main's folded activation lead frame ([994cf4f](https://github.com/labscommunity/cascadia/commit/994cf4f0683a87fdfaabbceaa72c3382717ef0e7))
* **shim:** a non-null under-sized buffer errors instead of degrading to a size query ([50ff844](https://github.com/labscommunity/cascadia/commit/50ff844c2dd2cd3583670819b9166446fbcfb57c))
* **shim:** dtype guard must not materialize a dynamic destination state ([b320fac](https://github.com/labscommunity/cascadia/commit/b320fac6f03ffb9a268173c0735acbe1a3988dde))
* **shim:** fail get_state_blob on an unmappable element type ([16e4a9b](https://github.com/labscommunity/cascadia/commit/16e4a9b16f70c0b5d22dd1e26aa571543ef47f07))
* **shim:** positional restore requires verbatim name equality per slot ([65a1e18](https://github.com/labscommunity/cascadia/commit/65a1e180854a6bbcb1233ebb5cbd7947cf96a9f0))
* **shim:** require the donor's element type to match the destination state's ([76de3ae](https://github.com/labscommunity/cascadia/commit/76de3aefaa5ec8ef6f92cc9d22f152b4ccef8ca6))
* **shim:** search for the Family-A KV alias instead of anchoring at position 0 ([14a3490](https://github.com/labscommunity/cascadia/commit/14a3490231f05b610e2ffc54a772d305f0522f42))
* **shim:** validate declared dims and dtype before constructing the restore tensor ([76f136e](https://github.com/labscommunity/cascadia/commit/76f136e01d3090672bda082641afbc30eb2e432b))
* **sparse-moe:** plane warm-resume could never commit; discarded prefill samples desynced the RNG ([2fc886b](https://github.com/labscommunity/cascadia/commit/2fc886b8c441c5dd96b26d4424f5692bf4378a38))
* **test:** gate the symlink-mirroring k26 test to unix ([fdab2f0](https://github.com/labscommunity/cascadia/commit/fdab2f0dce2675b533d59156b8ae4a0990b448be))


### Documentation

* **kv:** tighten warm-resume comments — terse, WHY once in kv_seq_from_blob ([94dae7e](https://github.com/labscommunity/cascadia/commit/94dae7e4bd0020564c9053973a40bc1b8683893b))
* **m2:** correct transformers pin for MiniMaxM2 export (&gt;=5.2,&lt;5.5, not 4.57) ([ae26faf](https://github.com/labscommunity/cascadia/commit/ae26faf8bab80cce3e03bdd7725f1933a45cc979))
* **shim:** set_state_blob matches by canonical identity, not position ([a09e218](https://github.com/labscommunity/cascadia/commit/a09e2189d38c6a22edc6635ec2fe8d33df7714d2))


### Testing

* **engine:** pin retryable frame-start timeout vs fatal idle ceiling ([7b9a0e2](https://github.com/labscommunity/cascadia/commit/7b9a0e29fddb1d00ebca374f1fa0f593521bd4cc))
* **k26:** native sparse-MoE fixture generator + load proof, fixture-gated ([2f56a85](https://github.com/labscommunity/cascadia/commit/2f56a8503398c4d739fc3332132e4a5078959b01))
* **ovmoe:** pin the captures-fallback H.1a residual on OvMoeKvHolder::export ([b9c877a](https://github.com/labscommunity/cascadia/commit/b9c877acd0d97d324264b4d569012eafa3f5145c))
* **wire:** golden the GetV2 field layout ([d0cf6e1](https://github.com/labscommunity/cascadia/commit/d0cf6e1a4c032ae175c464917721d4a3494c8da9))


### CI

* **ai-pc:** accept bash from PATH and document the Git for Windows prereq ([c810f66](https://github.com/labscommunity/cascadia/commit/c810f66ec4e6a3631e9fe37e41b8a6c5360a5321))
* **ai-pc:** fail fast when OpenVINO DLL dirs are missing or the var is unset ([0d40bae](https://github.com/labscommunity/cascadia/commit/0d40baee0069623010c1b626635099e081bfa567))
* **ai-pc:** fix Windows self-hosted reliability ([80a0fd7](https://github.com/labscommunity/cascadia/commit/80a0fd7a8e3e5ce1738049d12857acd35e6b1d50))
* **ai-pc:** fix Windows self-hosted reliability ([357be44](https://github.com/labscommunity/cascadia/commit/357be447008696552bfbbdb27da274a76792ba2f))
* **ai-pc:** move step rationale into comments per file convention ([7a572e9](https://github.com/labscommunity/cascadia/commit/7a572e9a944a723164ba987499c017a348c9f928))
* **ai-pc:** pass INTEL_OPENVINO_DIR via env instead of template interpolation ([a1cc1b6](https://github.com/labscommunity/cascadia/commit/a1cc1b67b767b2db9b1ad1e3bb5af12fe38bf3f1))
* check the whole workspace under kv_coord, not just the two engine crates ([c3304d0](https://github.com/labscommunity/cascadia/commit/c3304d03e522985b3b95372bf6bbed1fc238c25c))
* test the engine crates with --features kv_coord ([701ee77](https://github.com/labscommunity/cascadia/commit/701ee77394a4eabde1b4afc0abbcccc975c14880))


### Miscellaneous

* **kv:** remove CASCADIA_KV_DEBUG diagnostic instrumentation (issue-34) ([4adb2c1](https://github.com/labscommunity/cascadia/commit/4adb2c102dbba5eaa3d93863d4985a0f33af6431))
* **kv:** remove CASCADIA_KV_DEBUG diagnostic instrumentation (issue-34) ([7ee7196](https://github.com/labscommunity/cascadia/commit/7ee7196c1175d1b2463288a0800fd7357050fb4b))

## [0.2.0](https://github.com/labscommunity/cascadia/compare/v0.1.8...v0.2.0) (2026-08-18)


### ⚠ BREAKING CHANGES

* **glm5:** GlmRunner::load_staged now takes a StageOpts argument, and the default embed/lm_head precision is bf16 (set CASCADIA_GLM5_F32_HEAD for exact f32).

### Features

* **api:** accept chat_template_kwargs.enable_thinking (vLLM/SGLang convention) ([46b0fb9](https://github.com/labscommunity/cascadia/commit/46b0fb917cbc217b1ebef66f5c7a066787b5d6db))
* **api:** ChatPromptRenderer::render_with_opts exposes enable_thinking ([6440d24](https://github.com/labscommunity/cascadia/commit/6440d248464988d69f95e93d8a0a8d118e54df46))
* **api:** plumb reasoning_effort into the chat template ([744ffca](https://github.com/labscommunity/cascadia/commit/744ffcafdf759aab3117640e81d4cf707d9e30a1))
* **api:** smoke-render the chat template at load ([dbabf24](https://github.com/labscommunity/cascadia/commit/dbabf2437ac6183ff2edb3f51af2488d6ba7e418))
* **api:** streaming reasoning splitter for hybrid-reasoning models ([3338f63](https://github.com/labscommunity/cascadia/commit/3338f6365e5841ff09e29e34f1a29c7b5f2d8403))
* glm-5.2 support in the sparse-moe engine ([8268643](https://github.com/labscommunity/cascadia/commit/8268643b36408cebb356786784c34b971b1485b7))
* **glm5:** batch-union MoE kernel (dedup expert loads, bit-exact) ([a8436fe](https://github.com/labscommunity/cascadia/commit/a8436fea269e27504a0e3fcabc0c658518c4ea1d))
* **glm5:** batched prefill across the pipeline (StagedRunner::forward_layers_batch) ([208b654](https://github.com/labscommunity/cascadia/commit/208b654951f81f7c57cd29a073159d2780ae7cbc))
* **glm5:** batched prefill path (GlmModel::prefill uses batch-union MoE) ([c48a8e7](https://github.com/labscommunity/cascadia/commit/c48a8e712a24c0861162e42dde6ff33c22a0606b))
* **glm5:** cache prefix KV post-decode (prompt + response) ([36fd8d0](https://github.com/labscommunity/cascadia/commit/36fd8d0444fca5b406c3186b97dfbe0e5acf7099))
* **glm5:** CASCADIA_GLM5_ATTN_PROFILE — attention share of prefill ([c50f83e](https://github.com/labscommunity/cascadia/commit/c50f83e52b5dbfb453beb7f6344303676fbfd56e))
* **glm5:** CASCADIA_GLM5_NOPIN kill-switch for the residency A/B ([764acde](https://github.com/labscommunity/cascadia/commit/764acdeaa01286f46b0a9788aa625ab56ebeae41))
* **glm5:** chunk batched prefill into &lt;=256-row windows for long prompts ([96dccc9](https://github.com/labscommunity/cascadia/commit/96dccc92e047d2f0d5aaa732c3250c3e03d9f80b))
* **glm5:** default bf16 embed/lm_head; thread precision via StageOpts ([23f5685](https://github.com/labscommunity/cascadia/commit/23f56851ba6ef3a93a2c8ee533c204f2402297f1))
* **glm5:** distributed IndexShare via full-aligned layer split ([c328182](https://github.com/labscommunity/cascadia/commit/c32818268e1b22e2ca090857eeb5b99fca270b7e))
* **glm5:** distributed KV-prefix cache rank-0 driver (2b/2) ([f28930f](https://github.com/labscommunity/cascadia/commit/f28930f0af76d36d6aea141a637f58a389c098b3))
* **glm5:** DSA lightning indexer (raw-position top-k selection) ([dacf16c](https://github.com/labscommunity/cascadia/commit/dacf16c0632c952642a59711e8d4c886f562944a))
* **glm5:** export + load the DSA indexer per layer (end-to-end long ctx) ([9ce46f1](https://github.com/labscommunity/cascadia/commit/9ce46f1a509e856a1b2ef72b94813adb40b18379))
* **glm5:** export + load the MTP draft head (bf16) for spec decode ([88d7d5f](https://github.com/labscommunity/cascadia/commit/88d7d5f04938cac7525f6ebf66cb76fea8b04d7d))
* **glm5:** export_real emits the bf16 MTP draft head from FP8 ([113ca4c](https://github.com/labscommunity/cascadia/commit/113ca4cd2098a4c8a6feef2457e33b508b648d5a))
* **glm5:** exporter (config validation, manifest, int4 layout) ([89190be](https://github.com/labscommunity/cascadia/commit/89190bef5d32bc226d765a44fa737d8f206e464f))
* **glm5:** expose layer_split as the single source of truth ([b2cee73](https://github.com/labscommunity/cascadia/commit/b2cee730b87830d3eb59d141fd839775a4024735))
* **glm5:** full model + greedy-token parity (embed -&gt; layers -&gt; lm_head) ([60fc734](https://github.com/labscommunity/cascadia/commit/60fc734a29625520cc34ff764217296b6d872f27))
* **glm5:** full transformer layer (norms + attn + MoE, pre-norm) ([a18b5ae](https://github.com/labscommunity/cascadia/commit/a18b5aef13412909bf4aebc4eddd1637b371e2a6))
* **glm5:** glm5_run example — single-process greedy run harness ([b7234cf](https://github.com/labscommunity/cascadia/commit/b7234cfec920bb5db16f000302b2f2dec3b04b0a))
* **glm5:** glm5_shardcheck — verify M-rank pipeline == single-process ([e0db0dc](https://github.com/labscommunity/cascadia/commit/e0db0dcc477a3a5f2ad268956f274a6e42567934))
* **glm5:** glm5_spec bench example (mmap model + MTP spec decode) ([874a094](https://github.com/labscommunity/cascadia/commit/874a09436b29189ee0b18b0dc92bbf17c9f40fab))
* **glm5:** grammar-constrained decoding with forced-run batching ([9b99573](https://github.com/labscommunity/cascadia/commit/9b99573e0f5f2e0cbde7e7d96336048e9cd24087))
* **glm5:** IndexShare runtime — shared layers reuse the carried top-k (single-process) ([52973ae](https://github.com/labscommunity/cascadia/commit/52973ae66e102e8221e549f011a0c0e36771223a))
* **glm5:** int3 expert quant format (A2 foundation) — pack + dequant + tests ([759164c](https://github.com/labscommunity/cascadia/commit/759164c4b350e6dde26fc9f3b64120f87f6dac20))
* **glm5:** interleaved RoPE reused from the dsv4 shell ([47451c1](https://github.com/labscommunity/cascadia/commit/47451c1fb844caed71953d97e8111171a427abc4))
* **glm5:** KV snapshot/restore for prefix caching (spike) ([fc0f94d](https://github.com/labscommunity/cascadia/commit/fc0f94d533bdbae3d0aead3687d3a26d69734273))
* **glm5:** KvPrefixCache — reuse a shared prompt prefix (single-process) ([2ab7918](https://github.com/labscommunity/cascadia/commit/2ab79187dc46f4610cd9f605f58984b7389d1bef))
* **glm5:** learned-pin residency — record routing, mlock hot experts ([692338a](https://github.com/labscommunity/cascadia/commit/692338a6f5ad7c9c310819c90dcf73497737a3d3))
* **glm5:** loader + int4 numeric contract + round-trip parity ([ad2b993](https://github.com/labscommunity/cascadia/commit/ad2b993f954d937abd34b75868c61fc3bb999ccb))
* **glm5:** log KV-prefix cache hits (reused/prompt/suffix) ([1a939f6](https://github.com/labscommunity/cascadia/commit/1a939f6486c83f391dcdedb6a86d2957961870f0))
* **glm5:** MLA attention (classic V3) with absorbed-latent decode ([d16069f](https://github.com/labscommunity/cascadia/commit/d16069f203a32385d5fd6855883abcb033f8f125))
* **glm5:** mmap int4 expert path (loads the real model) ([f856794](https://github.com/labscommunity/cascadia/commit/f85679449b0a9f013ba0656442d09e6aeec270b7))
* **glm5:** MoE block (router + top-k experts + shared expert) ([d96fd3b](https://github.com/labscommunity/cascadia/commit/d96fd3beefabae58966c776e8a83e8c303dccc41))
* **glm5:** MoE router gate (sigmoid + noaux_tc) with golden harness ([42f99eb](https://github.com/labscommunity/cascadia/commit/42f99ebc1f17be2545cf4db69c9b0096be928cfc))
* **glm5:** MTP head (DeepSeek-V3 draft chain for speculative decode) ([87f7566](https://github.com/labscommunity/cascadia/commit/87f7566d8214c60836336e607d6bc932f75a8774))
* **glm5:** MTP speculative decode loop + KV rewind (single-process) ([cc9abb3](https://github.com/labscommunity/cascadia/commit/cc9abb3abe270d9c91dae32cd92a0c8457f446fd))
* **glm5:** opt-in bf16 embed/lm_head to grow the pin budget ([5228b25](https://github.com/labscommunity/cascadia/commit/5228b2590af33df8bdf935d207992ceaf16aa068))
* **glm5:** opt-in bf16 MLA KV cache to free long-context RAM ([4389716](https://github.com/labscommunity/cascadia/commit/43897166ca151e86a535834c121024d1d60e10fd))
* **glm5:** optional OpenVINO expert backend (iGPU / NPU / CPU) ([7caceb6](https://github.com/labscommunity/cascadia/commit/7caceb6e837fbc6f4d30b7b0df61a3b3d9831b2a))
* **glm5:** OV expert cache knobs config-first + effective-config log ([51d8ef3](https://github.com/labscommunity/cascadia/commit/51d8ef3845b0e1464d1781f4af89db311d764ef4))
* **glm5:** per-rank slice KV cache + GlmRunner snapshot/restore (Phase 2a) ([b6fbb0a](https://github.com/labscommunity/cascadia/commit/b6fbb0ac5a7234ec0dd2389858bd500d0915f71f))
* **glm5:** per-section decode profiler (CASCADIA_GLM5_PROFILE) ([aa518c9](https://github.com/labscommunity/cascadia/commit/aa518c9aa13909097aa9b6d9cb9ef456ed1a1ed4))
* **glm5:** per-token IO + residency telemetry in decode profiler ([be5ba5f](https://github.com/labscommunity/cascadia/commit/be5ba5fab7775de61e27e3dbc59b5a9390817c46))
* **glm5:** pipeline driver uses batched prefill (ForwardBatchPrefill frame) ([e8e56e5](https://github.com/labscommunity/cascadia/commit/e8e56e512ea11341ac698adca9a6f4e408120259))
* **glm5:** prefix-cache wire frames + worker handling + trait hooks (2b/1) ([5ab27eb](https://github.com/labscommunity/cascadia/commit/5ab27eb2985df40f50ea33f9d507f8df70ca276e))
* **glm5:** real FP8-&gt;int4 exporter (export_real) + validated round-trip ([d8b6038](https://github.com/labscommunity/cascadia/commit/d8b60387142c4ad323609f881e6ec4884eff4402))
* **glm5:** residency budget + routing histogram (learned-pin core) ([3627b32](https://github.com/labscommunity/cascadia/commit/3627b32bf1592aaebe043778d6a6c76af40a2315))
* **glm5:** resumable exporter (.done markers + disk pre-flight) ([ffd3eb1](https://github.com/labscommunity/cascadia/commit/ffd3eb14783e6e9c1f21a73a78a046fe84bc7f4c))
* **glm5:** router-lookahead expert prefetch (opt-in) ([42f963f](https://github.com/labscommunity/cascadia/commit/42f963f2f1c1b4787f6ec2bc65038eafc9028f5d))
* **glm5:** single-stage generate() batch-unions the prefill ([abb318d](https://github.com/labscommunity/cascadia/commit/abb318d45ce7d8451bafa1d0182aa0358377238b))
* **glm5:** staged GlmRunner + engine sniff (single-stage) ([fd59e26](https://github.com/labscommunity/cascadia/commit/fd59e268df33a613d908d760504d454dcc71ee84))
* **glm5:** tunables as builder config fields, env as fallback ([88c2244](https://github.com/labscommunity/cascadia/commit/88c224414b97ec868ed64d73e04cec7b6f718e57))
* **glm5:** wire DSA lightning indexer into MLA attention (long ctx) ([b201f71](https://github.com/labscommunity/cascadia/commit/b201f711d4a36990315cd7e1da88f0197c34a03b))
* **sparse-moe:** per-token streaming in PipelineEngine (glm5 + dsv4) ([4424e74](https://github.com/labscommunity/cascadia/commit/4424e740a8c7f7a787f840f570d797e11897e672))
* **tools:** tiny glm5 export gains a real indexer (T3 parity prerequisite) ([401c2d6](https://github.com/labscommunity/cascadia/commit/401c2d6d6013c0a0f1f2445784aa07f8f10a97fb))


### Bug Fixes

* **api:** arm tool-call scratchpad scan from the request, not sniffed text ([45a9b2c](https://github.com/labscommunity/cascadia/commit/45a9b2c984c5a77173080211a008ad681bdf5a0a))
* **api:** don't let a pre-engine 5xx latch pipeline readiness ([afa1257](https://github.com/labscommunity/cascadia/commit/afa1257d727cda329a7b98879c2aacbf250e8524))
* **api:** don't parse tool calls drafted inside the reasoning scratchpad ([8052976](https://github.com/labscommunity/cascadia/commit/8052976d55d4669af070fbde44a4d3cf5e34ceb4))
* **api:** drop a GLM tool call whose argument pairs are truncated ([d013ec0](https://github.com/labscommunity/cascadia/commit/d013ec00d68911e292cc13155232294f5bc4f90e))
* **api:** fail tool requests that cannot be templated ([3f805e3](https://github.com/labscommunity/cascadia/commit/3f805e303a25df381adb072dcfc4b7b0c3181ad8))
* **api:** map reasoning_effort onto GLM's two-level template vocabulary ([e515aea](https://github.com/labscommunity/cascadia/commit/e515aea1a9f5d47d45413e785a5a0989a48bac1b))
* **api:** parse GLM zero-argument tool calls instead of dropping them ([0ff0d56](https://github.com/labscommunity/cascadia/commit/0ff0d56bd8c7e77778b488f439e02d970a863069))
* **api:** parse GLM's arg_key/arg_value tool-call dialect ([97ab19f](https://github.com/labscommunity/cascadia/commit/97ab19ff3fe63bdda3617cbcce56cd19da4acd17))
* **api:** render chat templates with json.dumps semantics ([1007dd2](https://github.com/labscommunity/cascadia/commit/1007dd27cfa0da89a9cb53f0515ecb32d0814246))
* **api:** sanitize Jinja2 numeric dot-index for minijinja chat templates ([6749762](https://github.com/labscommunity/cascadia/commit/6749762e9772a59812d7ae4cc255c2165eb5ddab))
* **ci:** commit glm5 golden fixtures and delete the skip guards hiding them ([2f5ec34](https://github.com/labscommunity/cascadia/commit/2f5ec3439ad82806983900d9d675935579440b30))
* **dashboard:** 404 a missing SPA asset instead of serving the app shell ([7c84888](https://github.com/labscommunity/cascadia/commit/7c848887526e0d8b9bb0ea0b96d5b4f4c3bc0436))
* **dashboard:** honest / + embed the SPA in release bundles ([e7d54f8](https://github.com/labscommunity/cascadia/commit/e7d54f81c9b22a8306efe3a8277622d164f859c5))
* **dashboard:** pointer page at / and honest log when SPA is not embedded ([2af1889](https://github.com/labscommunity/cascadia/commit/2af188991abfe4bef3aba6bdfec9797deccfaa6e))
* **deps:** bump h2 to 0.4.16 for RUSTSEC-2026-0258 ([93b9574](https://github.com/labscommunity/cascadia/commit/93b9574deed1b097a3fb161c3eaa5360edb9e76f))
* **dist:** drop the connection when a token reply times out ([fae2173](https://github.com/labscommunity/cascadia/commit/fae217382635114969539e647bfce0571ce9f749))
* **engine-openvino:** fold seq and position into one unambiguous lead frame ([f483b91](https://github.com/labscommunity/cascadia/commit/f483b91bfaafb8249a1898c2d34030604a17dea1))
* **engine-openvino:** let a relay rank exit when its downstream stops answering ([10a1fce](https://github.com/labscommunity/cascadia/commit/10a1fce94e350d923ab4974e2b45361d40175c73))
* **engine-openvino:** NACK the upstream when a relay step fails after consuming ([1ef8f11](https://github.com/labscommunity/cascadia/commit/1ef8f1139e8e95780db1fc7ef64947b8d1b42c49))
* **engine-openvino:** report why a token wait gave up, and stop flooding on discards ([06c9e80](https://github.com/labscommunity/cascadia/commit/06c9e8016de746206cfe6e9fbb7ca8715bdf0c2e))
* **engine-openvino:** restore the widened prefill token budget ([269570d](https://github.com/labscommunity/cascadia/commit/269570d5540db4bd1b7300510a42d3431fff2515))
* **engine-openvino:** validate the token frame by dtype and shape, not length ([a283f82](https://github.com/labscommunity/cascadia/commit/a283f824230c92888ed6ebd9b08d57d48645b7ab))
* **engine:** per-hop seq echo to discard stale orphan tokens ([030d374](https://github.com/labscommunity/cascadia/commit/030d374ca37da1ca62cb752367078b0668519df0))
* **engine:** report prompt_tokens instead of always 0 ([94a19d1](https://github.com/labscommunity/cascadia/commit/94a19d1eb0bd3fda658ea8c52b78dfaac22b89c8))
* **glm5:** address OV-backend review findings + streaming n_tokens ([463d3f3](https://github.com/labscommunity/cascadia/commit/463d3f33300e2f6242bbcecbb29b16a7795a48c1))
* **glm5:** byte-budget the OV expert cache; typed exhaustion errors ([31bf016](https://github.com/labscommunity/cascadia/commit/31bf016bd613ee89f761509006e9adbd9928b130))
* **glm5:** correct DSA indexer to the real GLM-5.2 layout (names + IndexShare) ([ddd18b1](https://github.com/labscommunity/cascadia/commit/ddd18b121baddb1e1206cf6d1166e562b5dfb2ed))
* **glm5:** don't cache the KV prefix when finalizing a failed decode ([c0dff06](https://github.com/labscommunity/cascadia/commit/c0dff06d46d5666ac11edb1a2914687ca0ce6972))
* **glm5:** exporter carries chat_template.jinja + serving sidecars ([11c8f9f](https://github.com/labscommunity/cascadia/commit/11c8f9f4f741388f6e1526319ca6b037b912d398))
* **glm5:** gate mmap mlock/madvise(WILLNEED) behind cfg(unix) ([a632d91](https://github.com/labscommunity/cascadia/commit/a632d91beea3c6cded845e873ede7e1d48f50107))
* **glm5:** gate the nvme_readbench example to unix ([f41bb4c](https://github.com/labscommunity/cascadia/commit/f41bb4c8c4b98dfe29d89bfa7bdad6ad55c89eb1))
* **glm5:** guard cross-rank IndexShare split + bound context; fix pin budget ([4009d34](https://github.com/labscommunity/cascadia/commit/4009d3488c9e2ff62cfebfe55aed88e8d81db0a1))
* **glm5:** guard zero-layer ranks + fill each rank's pin budget ([40aea61](https://github.com/labscommunity/cascadia/commit/40aea61afa6a23d5cedeb510ed438a99dc2c1f74))
* **glm5:** honour restore_prefix's result instead of discarding it ([ccc09ae](https://github.com/labscommunity/cascadia/commit/ccc09aecbd84e568adf816062fa6610cf9d461f9))
* **glm5:** measure real expert-cache hit% via working-set probe ([2890dcb](https://github.com/labscommunity/cascadia/commit/2890dcbfdcb083a829db923389d3817f925e2b73))
* **glm5:** MLA latent-norm eps 1e-6 + exporter hardening (pre-run) ([cea167b](https://github.com/labscommunity/cascadia/commit/cea167b68a2bef6105c2d61bcc8ad5bf08dcaf30))
* **glm5:** OV expert exporter emits int4 (u4) IRs by default ([906b39c](https://github.com/labscommunity/cascadia/commit/906b39ce1ecd967940a2b35877d61dd00a496326))
* **glm5:** rate-limit OV stats dump; it contended with the decode hot path ([c8747a7](https://github.com/labscommunity/cascadia/commit/c8747a7b1b7cc912437bbf7fd9786da74d4117e0))
* **glm5:** reject an OV expert output of the wrong length ([5406e48](https://github.com/labscommunity/cascadia/commit/5406e48e9cc10133c6309a923748b7806b12f348))
* **glm5:** reuse the prefix key for an already-indexed sequence ([bfc59bd](https://github.com/labscommunity/cascadia/commit/bfc59bdbe20ad68f796c2726b3e723b290439bf1))
* **glm5:** route the remaining env switches through env_flag ([4d6655f](https://github.com/labscommunity/cascadia/commit/4d6655f42264b4e46f5e936d3ea75d77ff1ed665))
* **glm5:** stop FLAG=0 turning a feature ON ([88db932](https://github.com/labscommunity/cascadia/commit/88db9321e3e93391464a971927921bcded2646c4))
* **glm5:** stop OV expert accumulation from exhausting the iGPU pool ([195d584](https://github.com/labscommunity/cascadia/commit/195d5841d7ac1e70aa1ec6c3d1ab3e0f94024e7c))
* **glm5:** working-set governor keeps pressured nodes schedulable ([1d09747](https://github.com/labscommunity/cascadia/commit/1d09747f6f21ac7c8121e177b774b357f0a74bbe))
* **prefill-reply:** clamp the sparse-moe prefill budget under the frame-idle ceiling ([e1a4cdb](https://github.com/labscommunity/cascadia/commit/e1a4cdb01cc25338bdfd3ab50984d2ee2d82a67b))
* **prefill-reply:** restore the prefill token-wait budget on both pipeline paths ([0c242ba](https://github.com/labscommunity/cascadia/commit/0c242ba320f8a59dc5594c4b833ffd3773bd391d))
* **shim:** rebuild when the OV SDK is swapped behind a stable path ([739a658](https://github.com/labscommunity/cascadia/commit/739a65886f00f81f500a4504927833399dfcac9f))
* **tools:** emit MTP tensors in the fp8 fixture, don't lie in the manifest ([7be8087](https://github.com/labscommunity/cascadia/commit/7be808765d825a8e5aea0a45d9ee67b638e500e5))
* **tools:** gitignore the two new tiny-indexer fixture dirs ([596fc0a](https://github.com/labscommunity/cascadia/commit/596fc0af9cd4919d69b0d609d6885d741e1ec9bc))
* **tools:** glm5 expert OV IRs carry the bins' own int4 grid ([70456ef](https://github.com/labscommunity/cascadia/commit/70456efb40f58265b44db4d1c86c2943cce98979))
* **tools:** import os in the glm5 exporter ([34d785f](https://github.com/labscommunity/cascadia/commit/34d785fa107368b3ea6dac124a7b3dbcd7a03ec4))
* **tools:** refuse fp8 weights with no recognized block scale ([1f70891](https://github.com/labscommunity/cascadia/commit/1f708911a6567ab1090258df7bf3933185c86262))
* **transport:** bound every phase of the token recv, not just the frame start ([d592f11](https://github.com/labscommunity/cascadia/commit/d592f113a99c4c186e51fa937433f110c45630f3))
* **transport:** bounded non-fatal frame-start recv for active token responses ([cfe06aa](https://github.com/labscommunity/cascadia/commit/cfe06aa0edf12698a050e6b56ba38dc69f5f4694))
* **transport:** bounded non-fatal frame-start recv for active token responses ([09941ba](https://github.com/labscommunity/cascadia/commit/09941bacdbf509223c5f0b7a816c97a73187f76e))
* **transport:** classify every recv error explicitly, drop the wildcard ([53943f4](https://github.com/labscommunity/cascadia/commit/53943f4418a7063b3f08b7e5127575699e1be79e))


### Performance

* **glm5:** async lookahead expert prefetch (opt-in) + Windows autopin fixes ([e89358d](https://github.com/labscommunity/cascadia/commit/e89358dcf9c83fcdd7bc6ab5cc92ce8b5f72a58a))
* **glm5:** light R1 — explicit concurrent expert reads (opt-in) ([45d212e](https://github.com/labscommunity/cascadia/commit/45d212ec4c61b45ac4a917f5168bd5be4356a1c1))
* **glm5:** mlock always-active experts (shared + dense) — A1 ([81d4845](https://github.com/labscommunity/cascadia/commit/81d484509f094b5f3c027b2e92c329534deb67c7))
* **glm5:** native Windows mlock/madvise — VirtualLock + PrefetchVirtualMemory ([a4ddcbb](https://github.com/labscommunity/cascadia/commit/a4ddcbb8d5b9ec68a87369fcdd744e1bb39fd0bb))
* **glm5:** overlap expert reads with compute in the R1 path ([cf1c14c](https://github.com/labscommunity/cascadia/commit/cf1c14cf55bce462071ea8a398cd57b9200781f9))
* **glm5:** whole-expert WILLNEED readahead before each expert GEMV ([62714ce](https://github.com/labscommunity/cascadia/commit/62714ce7597e4539487936220a481db0aa9c8d99))
* **int4:** AVX-512 fused dequant+dot for the expert GEMV ([fe2adc1](https://github.com/labscommunity/cascadia/commit/fe2adc1169505babbb10e271e651347dbefda53c))


### Refactor

* **engine-openvino:** one seq counter, and no seq to echo until one arrives ([6bd6bab](https://github.com/labscommunity/cascadia/commit/6bd6baba900fc1e4d125e8b49c6a9dc29d73393d))
* **engine:** extract StagedRunner trait; Dsv4Engine -&gt; PipelineEngine&lt;R&gt; ([dd718e5](https://github.com/labscommunity/cascadia/commit/dd718e52457372a105dddef26565d2937166f5ee))
* **glm5:** AnyExpert enum for int4 expert storage (no behavior change) ([a4f3ee4](https://github.com/labscommunity/cascadia/commit/a4f3ee4fcabb76183dde7a11374278f93d95672f))


### Documentation

* alpha status; ci: manual dispatch for release PR ([5e90aa4](https://github.com/labscommunity/cascadia/commit/5e90aa42ee0b6e5656f8ab7723943c003c7b1946))
* bump status from pre-alpha to alpha ([39cae3f](https://github.com/labscommunity/cascadia/commit/39cae3f25d4feaab521a592dd229b8d2764ddbb3))
* correct the comments the rebase and these fixes falsified ([d0f51b5](https://github.com/labscommunity/cascadia/commit/d0f51b5a362a8fba27d133b848ba7e888374da2e))
* **dashboard:** only release builds truly embed the SPA ([8a361e6](https://github.com/labscommunity/cascadia/commit/8a361e62ac1065a8c67a8696f996ee94cb063104))
* document the web dashboard and its two-step embed build ([a075a8d](https://github.com/labscommunity/cascadia/commit/a075a8d0f75a75b67cf6c0a1584fc12b2a76123f))
* drop the unimplemented /api/events endpoint ([4649ae9](https://github.com/labscommunity/cascadia/commit/4649ae9968ea6c61dad67c9d73247834b4bb7c94))
* **glm5:** architecture + implementation status ([b7c523c](https://github.com/labscommunity/cascadia/commit/b7c523c1bea1d26ad1ca570d7f04fb4bd52c7ffa))
* **glm5:** correct the int4 capacity numbers against a real export ([bf66527](https://github.com/labscommunity/cascadia/commit/bf66527d76d662ca68abb604e88e1d6bf84d0160))
* **glm5:** correct the MTP dtype and the shipped-but-"deferred" entries ([8af9e30](https://github.com/labscommunity/cascadia/commit/8af9e3050694c9c38d9f3bb1db03582c84063262))
* **glm5:** describe the out-of-process consumer without naming it ([a7e2256](https://github.com/labscommunity/cascadia/commit/a7e22564bb77f56a7bf22e97eeedf3f768e55198))
* refer to the internal tracker without naming the private repository ([45278bb](https://github.com/labscommunity/cascadia/commit/45278bb5526b52ff5f11bc15e1aa40de23ee1d21))
* scope the dashboard release-bundle claim to bundles that have it ([9c83ca3](https://github.com/labscommunity/cascadia/commit/9c83ca38c18fa2df06a778f331627d26800ad120))


### Testing

* **engine-openvino:** cover the relay escalation decision ([c9a3173](https://github.com/labscommunity/cascadia/commit/c9a3173d707cf2651600ef10d1f17ef09630d9cf))
* **engine-openvino:** pin the frame-start classification against the real Display ([f2e1951](https://github.com/labscommunity/cascadia/commit/f2e1951df709e000df00122c9ba10417bbf5b9ac))
* **engine:** pin &lt;/think&gt; special:false survives skip-special decode ([51a995f](https://github.com/labscommunity/cascadia/commit/51a995f83e2f4f629ac0370638f887b947b1b440))
* **glm5:** 2-rank pipeline over real transport matches single-stage ([2f2da16](https://github.com/labscommunity/cascadia/commit/2f2da16bf411477415760fe54caea1f97eae3823))
* **glm5:** cover multi-window prefill over the real frames ([fe2de02](https://github.com/labscommunity/cascadia/commit/fe2de0299e93ed86b5c2d1914e532167b14daab0))
* **glm5:** fail assert_close on non-finite values ([6ac80e1](https://github.com/labscommunity/cascadia/commit/6ac80e16d42b74390c80a43b5f15d2c6a2c173fe))
* **glm5:** M-rank pipeline parity dry-run (middle-relay ranks) ([ff06e46](https://github.com/labscommunity/cascadia/commit/ff06e46350793a0fe970aba2d320bc70919609c6))
* **glm5:** nvme_readbench — R1 go/no-go microbench (mmap-fault vs explicit concurrent reads) ([ade2597](https://github.com/labscommunity/cascadia/commit/ade25976bdfbac7ea014abb659ac03044b6b0cae))
* **glm5:** pin the slice cache's key-reuse eviction contract ([cb6a5bd](https://github.com/labscommunity/cascadia/commit/cb6a5bd6b520d58fb6b2570bb94841cded9db540))
* **glm5:** take the prefill window from MAX_BATCH_COUNT ([5208c43](https://github.com/labscommunity/cascadia/commit/5208c4332daef0bebb2fae1a3a22be96ae1d1187))
* **runner:** cover the engine-lock release the bounded token wait exists for ([e8b2e7e](https://github.com/labscommunity/cascadia/commit/e8b2e7e792028e66730457aa8a1072aef22bce3c))


### CI

* build, lint and test the dashboard embed path on every PR ([9260088](https://github.com/labscommunity/cascadia/commit/926008864766d4c81e4276d5c9e433b48fbcdf78))
* make release PR manual dispatch ([20fdcb4](https://github.com/labscommunity/cascadia/commit/20fdcb4193ef84804ec2525e4cee3b649b97a76d))
* **release:** build and embed the dashboard SPA in release bundles ([4104762](https://github.com/labscommunity/cascadia/commit/410476275ea3c901aab432f2fd6f63a26f0ffded))
* **release:** build the dashboard SPA in an isolated job ([637b5a5](https://github.com/labscommunity/cascadia/commit/637b5a5b11fb0d16786346c7580de0cbcca6bf84))
* restrict release-please dispatch to main ([58a2a07](https://github.com/labscommunity/cascadia/commit/58a2a078d452f0e6c38c310e1fb11be1af7538c7))


### Miscellaneous

* drop cross-repo issue refs; ignore attn-ov parity fixtures ([cb0e4ed](https://github.com/labscommunity/cascadia/commit/cb0e4ed151c37c1cf778ba37a4fa5e368fa453e4))
* **dsv4:** vendor the OV expert exporter into tools/ ([e0cfbcb](https://github.com/labscommunity/cascadia/commit/e0cfbcb26b900bfffe451e83c4bf7f914b861eb5))

## [0.1.8](https://github.com/labscommunity/cascadia/compare/v0.1.7...v0.1.8) (2026-08-11)


### Features

* **cli:** allow --packed-slots with --total &gt; 1 ([#122](https://github.com/labscommunity/cascadia/issues/122) fixed) ([1f55d68](https://github.com/labscommunity/cascadia/commit/1f55d681e45071c6dd079007f27cb54a9f5b3db7))


### Bug Fixes

* **cli:** reject --packed-prefix with --total &gt; 1 ([e93a394](https://github.com/labscommunity/cascadia/commit/e93a3944f04c29d09daa55c712762bb69ac0d769))
* **cli:** use generate_async in the stdin loop ([c464538](https://github.com/labscommunity/cascadia/commit/c46453860f240f54951a949adaad7cd653581add))
* **deps:** upgrade lru to 0.18 (RUSTSEC-2026-0253) ([740c88c](https://github.com/labscommunity/cascadia/commit/740c88ce154e295b95764887cd89333c8767cc74))
* **deps:** upgrade lru to 0.18 (RUSTSEC-2026-0253) ([6728209](https://github.com/labscommunity/cascadia/commit/6728209fb169c103416189a3f04d41420eebcd7c))
* **engine:** add BatchAborted error variant so NACKs can't be misclassified as connection-fatal ([af1e8e2](https://github.com/labscommunity/cascadia/commit/af1e8e2ca0730aa23543a8b55ef106a895e47eb7))
* **ov-runtime:** fail fast and loud after the packed downstream link is poisoned ([64a28b4](https://github.com/labscommunity/cascadia/commit/64a28b42964eb6308513d871e01046d577e5a709))
* **ov-runtime:** harden the packed multi-stage wire against loss and silence ([#122](https://github.com/labscommunity/cascadia/issues/122)) ([3b0eefd](https://github.com/labscommunity/cascadia/commit/3b0eefd1244e7fa5ba05617f6253afd7ba8c2b9f))
* **ov-runtime:** surface the step error when a NACK send also fails ([8fec1a1](https://github.com/labscommunity/cascadia/commit/8fec1a1819097821814fda795a2ec9357f9aa3ee))
* **runner:** apply deferred cancels on lock acquisition and bound the queue ([984dbcd](https://github.com/labscommunity/cascadia/commit/984dbcd069e229a9895b1a2cd2e243e12352edd2))
* **runner:** never block tokio workers on the engine mutex ([#122](https://github.com/labscommunity/cascadia/issues/122)) ([48047c7](https://github.com/labscommunity/cascadia/commit/48047c73751e9d10dcdbdd69e145fdcaa0aeb8bb))
* **runner:** wake parked streams even when a step panics ([84b81ab](https://github.com/labscommunity/cascadia/commit/84b81ab54ade8689c509bbcf4788859f6efda10a))
* **runner:** wake parked streams on submit's NotLoaded early return ([628ef9b](https://github.com/labscommunity/cascadia/commit/628ef9b3b22b1b98c4a86ace577350b38d605e50))


### Refactor

* **runner:** enforce wake-on-release via an engine lock guard ([3633f22](https://github.com/labscommunity/cascadia/commit/3633f22fcf949d6ab9fc4417790552981a9133cc))


### Documentation

* correct lock-protocol and blocking-API comments ([d2f5e16](https://github.com/labscommunity/cascadia/commit/d2f5e168ed42c6ec5323a9dbc192fb60488429c8))
* multi-stage packed is available again; record the [#122](https://github.com/labscommunity/cascadia/issues/122) root cause ([da8a892](https://github.com/labscommunity/cascadia/commit/da8a892c1f381e9b2e0ff17678e7122ba982e304))


### Testing

* **runner:** cover NACK-driven batch aborts and close-while-parked ([d157382](https://github.com/labscommunity/cascadia/commit/d1573820655501640f8af39ad19793ed27c3cfd7))

## [0.1.7](https://github.com/labscommunity/cascadia/compare/v0.1.6...v0.1.7) (2026-08-06)


### Features

* **metrics:** Prometheus /metrics endpoint — request, generation, engine, and transport metrics ([#16](https://github.com/labscommunity/cascadia/issues/16)) ([adcd3cd](https://github.com/labscommunity/cascadia/commit/adcd3cdcb2acee3e27e15819306acd1104410440))
* **metrics:** Prometheus /metrics endpoint — request, generation, engine, transport metrics ([#16](https://github.com/labscommunity/cascadia/issues/16)) ([fa0a326](https://github.com/labscommunity/cascadia/commit/fa0a3268c236118b4bb248d888dd4d694f80f295))


### Bug Fixes

* **engines:** stop reporting engine failures as empty successful completions ([a895407](https://github.com/labscommunity/cascadia/commit/a8954073ddc4f3155ddb99586b0475e13f00538f))
* **metrics:** count an over-window prompt rejection ([9ec49e7](https://github.com/labscommunity/cascadia/commit/9ec49e757494b2f16a37573a6ff5d9f588cecc4c))
* **metrics:** count every pre-generation engine rejection, not two of them ([9b67680](https://github.com/labscommunity/cascadia/commit/9b6768007d50bd148355a230fa73980d5e98f4ad))
* **metrics:** review round — cancel accounting, teardown, QueueFull capacity, timing artifacts ([d9b1f8f](https://github.com/labscommunity/cascadia/commit/d9b1f8fd1ca94f8afb38e1cc10290dda1c9756e9))
* **ov-genai:** a failed generate() must emit an error chunk, not an empty success ([14b6d49](https://github.com/labscommunity/cascadia/commit/14b6d49875adcced70bf96cedf4a48fd5ed8e998))
* **runner:** fail loud on shutdown, and book teardown deterministically ([bcf5a11](https://github.com/labscommunity/cascadia/commit/bcf5a118d4cd04875c9d74c5d8ecb367c60a47bd))


### Documentation

* **metrics:** fix the scrape example and the HELP strings that contradict their docs ([48b98a3](https://github.com/labscommunity/cascadia/commit/48b98a36b5e8c09bfbb64e6bee231b8b935cdcb8))


### Testing

* cover the teardown HTTP surface and the qwen36 empty-prompt rejection ([a503ad6](https://github.com/labscommunity/cascadia/commit/a503ad6be68d4de47b65b00aa32eac6eeb7ed343))
* **metrics:** make three assertions that could not fail actually fail ([8c10cbf](https://github.com/labscommunity/cascadia/commit/8c10cbf1e7281162feaa164fcf2ad77c378dd083))
* **runner:** pin metric attribution across concurrent streams ([b75d4a3](https://github.com/labscommunity/cascadia/commit/b75d4a32019141574ecebf23ab30e96bd33553bf))

## [0.1.6](https://github.com/labscommunity/cascadia/compare/v0.1.5...v0.1.6) (2026-08-04)


### Features

* **npu:** continuous batching on the NPU via packed multi-slot decode ([7cd3190](https://github.com/labscommunity/cascadia/commit/7cd3190dab7c0afd814b077d121d6b45d26d3c05))
* **npu:** multi-stage packed wire + per-slot cancel ([63e44ee](https://github.com/labscommunity/cascadia/commit/63e44eed1da13868ad23a20e7fdcc09d3f57cebb))
* **npu:** packed multi-slot substrate for continuous batching (seq-as-batch) ([f36c408](https://github.com/labscommunity/cascadia/commit/f36c40884d462d06429e29027d674abe3b5dec69))
* **npu:** prefix caching via a shared read-only KV region ([14c9903](https://github.com/labscommunity/cascadia/commit/14c9903fb14704c2e5c1943abd1954f9276e4b3a))
* **npu:** wire packed multi-slot execution into OvRuntimeEngine ([9ab6746](https://github.com/labscommunity/cascadia/commit/9ab6746dd0b34fa3ab09ec7430c230d95c6c6908))


### Bug Fixes

* **dist-spec:** remove a byte-offset slice that could panic the worker ([6ba63b6](https://github.com/labscommunity/cascadia/commit/6ba63b66a93e9b9ff1dbe9ca5baa8112d31c0570))
* **npu:** answer 413, not 503, when a prompt cannot fit a packed slot ([165fcdd](https://github.com/labscommunity/cascadia/commit/165fcdd131b2c051fe84947931975a07936f32cd))
* **npu:** bound the packed multi-stage reply wait ([6560dda](https://github.com/labscommunity/cascadia/commit/6560dda6f79f10026ba485c07caadd89af5057e9))
* **npu:** complete packed prefill inside one step + report prompt_tokens ([481174b](https://github.com/labscommunity/cascadia/commit/481174b4be8f835e15354c7741985ee632f631ae))
* **npu:** keep attention sinks when a packed slot's KV region slides ([62efd11](https://github.com/labscommunity/cascadia/commit/62efd111ffc4484d2acf98b9569fbba537349947))
* **npu:** packed usage double-count + f16 wire dtype; document parity findings ([866544a](https://github.com/labscommunity/cascadia/commit/866544a158f243f5b536c8e01fd808693ced221f))
* **npu:** packed wire must use the block_in_place-aware dispatch ([126b232](https://github.com/labscommunity/cascadia/commit/126b23255f640269d0be1f1bb0398a973a42d666))
* **npu:** refuse a packed prompt that cannot fit its slot's KV region ([fee9c33](https://github.com/labscommunity/cascadia/commit/fee9c337bd6a3e8f6373fdc9c1d30bf7c85e5815))
* **npu:** refuse a packed variant narrower than its own slot count ([c5e64c7](https://github.com/labscommunity/cascadia/commit/c5e64c76eed7a977bd082139034fd8e13b301870))
* **npu:** withhold multi-stage packed — it can lose a token frame and wedge ([db18d0d](https://github.com/labscommunity/cascadia/commit/db18d0d37ecd516482f61d99677f4a607ed217e2))
* **runtime:** keep attention sinks when the single-task KV ring slides ([feff6b1](https://github.com/labscommunity/cascadia/commit/feff6b1c9a8249f19de3a09b0265a071deb80b76))
* **runtime:** stop the ov-runtime delta duplicating on a resolved glyph ([428a875](https://github.com/labscommunity/cascadia/commit/428a875311da23cb288ab223bf16f2c2f8b0497e))


### Performance

* **npu:** skip the prefill-variant compile in packed mode + NPU e2e results ([205f894](https://github.com/labscommunity/cascadia/commit/205f894fa715d98811b471622f873d94394498db))


### Documentation

* **npu:** fix stale plan-frame shape in perf doc — [1,3,S], not [1,2,S] ([0435285](https://github.com/labscommunity/cascadia/commit/04352854418ca4653d55d33f844a825bca00bbdb))
* **npu:** point the multi-stage gate at its tracking issue ([637cb96](https://github.com/labscommunity/cascadia/commit/637cb9678a4729214d995458afbc374d8e51e490))
* **npu:** reconcile packed-slots with [#116](https://github.com/labscommunity/cascadia/issues/116) continuous batching ([1ef825f](https://github.com/labscommunity/cascadia/commit/1ef825f440c04af9c57829951c0fe60a1c2997f6))
* **npu:** record the packed-slots end-to-end run ([1bd4009](https://github.com/labscommunity/cascadia/commit/1bd400949fbe5754bc6e4abcaa05a3b2697b495f))


### Testing

* **npu:** add solo-only capture mode ([8f0efe4](https://github.com/labscommunity/cascadia/commit/8f0efe477d91ecb5ca937c4ca297aa9ed8462a1b))
* **npu:** end-to-end accuracy parity harness for packed multi-slot decode ([3a034a9](https://github.com/labscommunity/cascadia/commit/3a034a9f6b7eb2aed5db9a287f5120e3e79fa5bc))
* **npu:** scored long-form accuracy benchmark ([6eff7c3](https://github.com/labscommunity/cascadia/commit/6eff7c31b4ea779df56f476ec782239e190748f0))
* **npu:** scored task-accuracy benchmark for packed decode ([10a76f0](https://github.com/labscommunity/cascadia/commit/10a76f012b3992969053012f1ac9d6d5103ceca5))

## [0.1.5](https://github.com/labscommunity/cascadia/compare/v0.1.4...v0.1.5) (2026-07-30)


### Bug Fixes

* **shim:** explicit bool return type in AVX-VNNI probe lambda ([828b401](https://github.com/labscommunity/cascadia/commit/828b401a45a414722f1d89c7a181008fd66b154d))
* **shim:** explicit bool return type in AVX-VNNI probe lambda ([1cebab4](https://github.com/labscommunity/cascadia/commit/1cebab4d9e96c27fb100cce0918337869772a780))

## [0.1.4](https://github.com/labscommunity/cascadia/compare/v0.1.3...v0.1.4) (2026-07-29)


### Features

* **cli:** warn when --cb is enabled on a CPU device ([c8b6c63](https://github.com/labscommunity/cascadia/commit/c8b6c63fa64915916b6f5a3ff4cdd8874872577a))
* **ov-genai:** continuous batching via ContinuousBatchingPipeline ([#20](https://github.com/labscommunity/cascadia/issues/20)) ([2229af0](https://github.com/labscommunity/cascadia/commit/2229af0576d0d82caec279ac0b3eb3033064779d))


### Bug Fixes

* **cli:** reject --cb on an NPU device ([280e1b5](https://github.com/labscommunity/cascadia/commit/280e1b5ff11212f6ed0cdb410db56635b6e4fe90))
* **ov-genai:** bound the cb liveness heartbeat so a wedged batch still fails ([2a83442](https://github.com/labscommunity/cascadia/commit/2a83442649ac2df4b6f04c4f710895abbe2b56c6))
* **ov-genai:** cb step() signals liveness so long prefills survive the stall guard ([c7f14f1](https://github.com/labscommunity/cascadia/commit/c7f14f12768262a0f14ad7eb44ee9bac2a072b95))
* **ov-genai:** cb warmup reports failure instead of always logging ok ([6420243](https://github.com/labscommunity/cascadia/commit/642024311a0480154050b91852b4ca938fabd2ba))
* **ov-genai:** report a cb scheduler eviction as an error, not a clean stop ([48a342b](https://github.com/labscommunity/cascadia/commit/48a342b0ddffc8703ca8d06fa1d0bd8cb725e191))
* **ov-genai:** stop admitting work onto a dead cb pipeline ([3b2812c](https://github.com/labscommunity/cascadia/commit/3b2812c73fe1186d32154b5d6d4cc63d98e046c2))
* **runner:** deliver a stream's chunks in the order the engine produced them ([abccce5](https://github.com/labscommunity/cascadia/commit/abccce53e677d5356ff173ae0db4873fc5c68197))
* **shim:** apply the chat template on the cb path ([88a3a76](https://github.com/labscommunity/cascadia/commit/88a3a762df7c4f61d56e1f4523505c5509b9a8dc))
* **shim:** move the cb UTF-8 hold-back to Rust and make it resync ([e24e2c8](https://github.com/labscommunity/cascadia/commit/e24e2c8d2605e9dc68d7a7454e2a75ba19581b35))


### Refactor

* **shim:** drop the unused cb has_unfinished entry point ([13320a4](https://github.com/labscommunity/cascadia/commit/13320a4e4301a444026a69c36a9c66dd0eb13c57))
* **shim:** tie a CbHandle's lifetime to its pipeline via Arc ([da219dd](https://github.com/labscommunity/cascadia/commit/da219dd3ea90d76950d6f2a01853def3da2e5fe0))


### Documentation

* **ov-genai:** characterise when --cb helps and when it hurts ([b55c754](https://github.com/labscommunity/cascadia/commit/b55c754df44d4aaade7ea042259c14b594bd9e9b))
* **ov-genai:** correct the --cb example and three overstated claims ([951a13b](https://github.com/labscommunity/cascadia/commit/951a13b39ad65f3bf28778e3ab324c632818a5aa))
* **shim:** correct the cancel-vs-stop and request_id claims ([8e1c824](https://github.com/labscommunity/cascadia/commit/8e1c8246deff30f8f2314f06c6a62e094c6a4cf3))


### Testing

* **ov-genai:** cover the cb engine state machine behind a pipeline seam ([f28dad0](https://github.com/labscommunity/cascadia/commit/f28dad0f0b8bb1035d969bc406124e5e51933f31))
* **ov-genai:** pin cb/LLMPipeline chat-template parity ([cb34bb9](https://github.com/labscommunity/cascadia/commit/cb34bb9d8078de600fef73dfa999a1a4139b2f2f))
* **shim:** cover the cb resync path with unit tests ([b06158d](https://github.com/labscommunity/cascadia/commit/b06158d5462101a5c9358fd9eae920ea50eafff7))

## [0.1.3](https://github.com/labscommunity/cascadia/compare/v0.1.2...v0.1.3) (2026-07-25)


### Features

* **cli:** --prefill-device / --no-chunked-prefill + shard --static-prefill-seq ([46c81b6](https://github.com/labscommunity/cascadia/commit/46c81b66bde8eece77d8a1b4e7500a242ffeb431))
* **engine:** --park-prefill — release prefill weights between prefills ([381c08b](https://github.com/labscommunity/cascadia/commit/381c08b31d1c0e57d01ba3b2c36f7bb32404855e))
* **engine:** chunked multi-token prefill + per-phase device on the static path ([cff6933](https://github.com/labscommunity/cascadia/commit/cff6933eca57dceffbddbdaebd5dd8aa05ff94c4))
* **engine:** consume AOT NPU blobs — .blob sibling of the IR imports instead of compiling ([638afa9](https://github.com/labscommunity/cascadia/commit/638afa9bc8ff97d72fb2ddfa1909ec81d4e10aa1))
* **engine:** npuw bank probe + park-without-cache warning ([2fb27d4](https://github.com/labscommunity/cascadia/commit/2fb27d48e1479460a11e6f12c542f93fd6a981f1))
* **export:** emit chunked-prefill static IR variant (--static-prefill-seq) ([c1482cb](https://github.com/labscommunity/cascadia/commit/c1482cbf00234c3ec5be613a7fcc4d1c0b16f431))
* hybrid NPU+CPU execution — chunked prefill on one device, decode on another ([ae008f8](https://github.com/labscommunity/cascadia/commit/ae008f8d8f1fdfc1f9c77fc401b867b31af22090))
* **shim:** AOT blob-import FFI + probe — NPU compile spike moves off-box ([6e97af2](https://github.com/labscommunity/cascadia/commit/6e97af2ea1dfc18958addb81791990befa9f1d1a))
* **shim:** AVX2 GEMV path, CSE weights-tag, residency probe + spike notes ([df9a770](https://github.com/labscommunity/cascadia/commit/df9a770354965e6254f43b568383a25a797bfb35))
* **shim:** CascadiaInt4Gemv extension op — decode GEMV from the .bin mmap (spike) ([4274b5d](https://github.com/labscommunity/cascadia/commit/4274b5d56b43157464bd368f8ba0ba322f0f8fed))
* **shim:** oneDNN embedding probe — endgame closed by data (fork-only kernels) ([ccb47b0](https://github.com/labscommunity/cascadia/commit/ccb47b0eee25d06dcb1d0f869411d457e020ed36))
* **shim:** PERF_COUNT profiling FFI + flat kernels — gap fully attributed ([487c5fb](https://github.com/labscommunity/cascadia/commit/487c5fb4f8b9ad16e4773c08ecd053f473286a31))
* **shim:** sibling GEMV fusion (q/k/v, gate/up) — measured perf-neutral ([e93a3e2](https://github.com/labscommunity/cascadia/commit/e93a3e23ea279ce3b7ce6fff7876065b50f10340))
* weight-residency paths for the hybrid split — --park-prefill, NPUW bank probe, in-place GEMV RFC ([7f4f37a](https://github.com/labscommunity/cascadia/commit/7f4f37ac5196d1d55e9c578d99f952af5f71aa73))


### Bug Fixes

* **engine-openvino:** lower the near-tie guard to the first token (10 -&gt; 1) ([c87c09d](https://github.com/labscommunity/cascadia/commit/c87c09d793e92aea6bbe231ecb4f69a98883cd27))
* **engine:** review findings — park on cancel/failure, parked-check before ring mutation, warmup ensure, doc placement, probe resilience, 1-based labels ([44a78fe](https://github.com/labscommunity/cascadia/commit/44a78feeb4b371bbfa346e1ff6af9b7cbbe8017d))
* **engine:** review findings — per-device prefill plugin props, cross-stage geometry check, loud argmax on truncated logits ([4bd6a51](https://github.com/labscommunity/cascadia/commit/4bd6a519166a32c5f8bc50e225b017f1c5b3503f))
* **engine:** review findings — window-parity cap, sub-chunking, guards, reuse ([fb5d9aa](https://github.com/labscommunity/cascadia/commit/fb5d9aaeb1e308fa80e78e65f4c351b192ec92ef))
* **engine:** warn on an unverifiable AOT blob instead of silently trusting it ([701cbf4](https://github.com/labscommunity/cascadia/commit/701cbf48012128d3e8f23f49cb713f618445c6ee))
* **export:** remove stale prefill variant on re-export ([5814e59](https://github.com/labscommunity/cascadia/commit/5814e599f95696104b8f314206b9497fe413be83))
* **review:** gemv CACHE_DIR strip via props, stale-blob guard, Linux SIMD/tbb build, matcher+fusion guards, tellg check, pacing safety, profiling truncation, xfer wait bug ([506c254](https://github.com/labscommunity/cascadia/commit/506c2540aa7ec9b348df9ee7f5997d2b45a8c667))
* **shim:** reject an empty CascadiaInt4Gemv weights_tag (CSE-merge guard) ([4177b9d](https://github.com/labscommunity/cascadia/commit/4177b9da2b1fbf34de196d0fcec64a568f107181))
* **transport:** pace large tensor payloads into bounded bursts — DERP-relayed links drop ~750KB single bursts intermittently ([8ec831a](https://github.com/labscommunity/cascadia/commit/8ec831a076dee25820b78f464539409fc66b53e3))
* **transport:** warn on an unparseable CASCADIA_SEND_BURST_BYTES instead of silent OFF ([9be0d40](https://github.com/labscommunity/cascadia/commit/9be0d409f03e550c9375c8e3b13fd20564ddf46a))
* **transport:** warn when CASCADIA_SEND_BURST_BYTES is clamped up to the 64 KiB floor ([566cdb5](https://github.com/labscommunity/cascadia/commit/566cdb5f94f086cf063857a50d399195d62961d0))


### Performance

* **shim:** gemv-offload 66-73% -&gt; 75-79% of stock; frontier mapped ([25d4133](https://github.com/labscommunity/cascadia/commit/25d413350296433461dbcdf0ecbd8b60db00dfae))


### Documentation

* **experiments:** track the gemv-offload spike notes incl. NPU cache-import init warning ([f49dcaf](https://github.com/labscommunity/cascadia/commit/f49dcaf1c691254d7cd656febead26941f5e1a3a))
* **experiments:** track the NPUW weights-bank probe notes (gitignore exempts) ([dff68e0](https://github.com/labscommunity/cascadia/commit/dff68e06960f14acf489997538fb09b4e5f6817a))
* **perf:** 70B blocker dossier — frame black-holing on DERP links, everything else ruled out ([eed8d82](https://github.com/labscommunity/cascadia/commit/eed8d82744fd50259bd070048ed2f94acf1ef6ff))
* **perf:** add 2-stage hybrid pipeline smoke numbers ([6db3ccd](https://github.com/labscommunity/cascadia/commit/6db3ccd87e2fc04d7ab3a65d6ceef713d4e555d9))
* **perf:** big-model NPU routes — 2-stage PP, AOT blob import, NPUW folding (all measured) ([d3ea8ce](https://github.com/labscommunity/cascadia/commit/d3ea8ce1635475018065007ef41dfcb313f51299))
* **perf:** correct over-window semantics + short-prompt and placement notes ([62000f5](https://github.com/labscommunity/cascadia/commit/62000f5ca6123ff8401f9f7651fbee065690ceb4))
* **perf:** device x model-size matrix (1B/3B/8B x CPU/GPU/NPU) on LNL 32GB ([fbd8970](https://github.com/labscommunity/cascadia/commit/fbd8970fef2a83ea085220f77bdb7536fbf9167d))
* **perf:** finalize tier results — 32B 3-box measured, 70B health-validated with isolated blockers ([0f80217](https://github.com/labscommunity/cascadia/commit/0f80217c2a0e96874808128f05818d5afac5e9a7))
* **perf:** hybrid NPU+CPU phase split — design, quickstart, measured results ([328ef7e](https://github.com/labscommunity/cascadia/commit/328ef7e6021e8866a199fb6e12d93f1134340122))
* **perf:** link PR [#107](https://github.com/labscommunity/cascadia/issues/107) in status line ([85cc925](https://github.com/labscommunity/cascadia/commit/85cc9256594271aaa5f20b0a97da76654e328f2c))
* **perf:** NPU TTFT tier benchmarks — method, 14B results, attribution, fleet deployment learnings ([08369df](https://github.com/labscommunity/cascadia/commit/08369dfa6f2669eebd55c39011be9ebe5b8814b6))
* **perf:** parking measurements + npuw bank probe results ([d15484a](https://github.com/labscommunity/cascadia/commit/d15484a6bd8859814e58b1ec961ceaf6b0ae3080))
* **perf:** post-fix re-validation numbers (33x long-prompt, over-window leg, warm smoke) ([0610d22](https://github.com/labscommunity/cascadia/commit/0610d2225237307d599993caea3f8051d794fc41))
* qualify the remaining token-exactness claims for near-tie tolerance ([aeba7eb](https://github.com/labscommunity/cascadia/commit/aeba7eb422e8c24db531a7328cd12f928e704fb0))
* **rfcs:** narrow the in-place GEMV ask per spike evidence ([8f1e37a](https://github.com/labscommunity/cascadia/commit/8f1e37a75b5f53eb3d2e7c1831cf8e0190138c98))
* **test:** parking leg is near-tie-tolerant parity, not token-identical ([527c9bc](https://github.com/labscommunity/cascadia/commit/527c9bca60fcb20b38f80ad83403a1995dd9e81f))


### Testing

* **cli:** cover shard --static-prefill-seq validation + forwarding ([4e6d109](https://github.com/labscommunity/cascadia/commit/4e6d1094222a75d945e4e1d029fa0e2584b9e2db))
* **cli:** cover the worker phase-split flag guards ([619a3be](https://github.com/labscommunity/cascadia/commit/619a3be7544b123899e8e34dcd3e49b98db172aa))
* **engine-openvino:** CASCADIA_PARITY_SOFT knob for GPU/cross-device sweeps ([64d9f93](https://github.com/labscommunity/cascadia/commit/64d9f93eb8579085b5b4d6968f4f6647197020fe))
* **engine-openvino:** CASCADIA_STATIC_TASKS knob — steady-state TTFT past the NPU cache-import init ([109603e](https://github.com/labscommunity/cascadia/commit/109603e84b12314f2d7e27c0b202327d9841e4a4))
* **engine-openvino:** CASCADIA_WARM_INFER — isolate graph-specific first-inference pathologies ([db06dd5](https://github.com/labscommunity/cascadia/commit/db06dd5c80a0d772c741556184fa569e0f542b64))
* **engine-openvino:** cover import_plugin CACHE_DIR strip ([d2dba1c](https://github.com/labscommunity/cascadia/commit/d2dba1c973e227aa2fc57bcbc9d8b45ce3d1636c))
* **engine-openvino:** parking leg honors CASCADIA_PARITY_SOFT ([ca937a0](https://github.com/labscommunity/cascadia/commit/ca937a0400e61764e59cb1274ae450426761726a))
* **engine-openvino:** sequential cache-warm probe — no overlapping NPU compile transients at pipeline bring-up ([988b1b9](https://github.com/labscommunity/cascadia/commit/988b1b90fad52e3fa7d5eed2a1215cd2c6f6554c))
* **engine-openvino:** tolerate near-tie prefill forks; correct the token-exact claim ([7623bbb](https://github.com/labscommunity/cascadia/commit/7623bbb000a2280dbb01ce9a8483a35d3f7d58dd))
* **engine-openvino:** unit-test the parity verdict (no hardware) ([968e2c4](https://github.com/labscommunity/cascadia/commit/968e2c406f8bbfcf56d722b95083157b7757b047))
* **engine:** chunked-prefill ring equivalence + phase-split parity gate ([f65701b](https://github.com/labscommunity/cascadia/commit/f65701bbcdeee3018089094eb2f77c3dcbcd5958))


### Miscellaneous

* **transport:** default send pacing off — did not resolve the DERP frame loss; kept as experiment knob ([4592fe2](https://github.com/labscommunity/cascadia/commit/4592fe2e8032b8129259e43fd55dfe5828b13079))

## [0.1.2](https://github.com/labscommunity/cascadia/compare/v0.1.1...v0.1.2) (2026-07-21)


### Features

* **api:** real /health readiness (was a hardcoded 200) ([c9b6ebc](https://github.com/labscommunity/cascadia/commit/c9b6ebc9683c18bb2501f156c4106790af9f4bf3))
* **cli:** serve a clean model name (basename of --model, or --served-model-name) ([0895c2d](https://github.com/labscommunity/cascadia/commit/0895c2d7faaa7484721b182c1c828bc278ef818f))
* deepseek v4 support in the sparse-moe engine ([4651129](https://github.com/labscommunity/cascadia/commit/46511292414eb677b81023bcff2d9b331c40298c))
* **dsv4:** DeepSeek-V4-Flash exporter + CPU reference model ([660e56a](https://github.com/labscommunity/cascadia/commit/660e56a761afa56d50a3b106638e210cb4902a4b))
* **dsv4:** optional OpenVINO int4 expert backend (GPU/CPU/NPU) ([7ee21a0](https://github.com/labscommunity/cascadia/commit/7ee21a09e46168186101d8ea071d10aab4bac2d5))
* **dsv4:** ship R1 chat_template so chat completions render instruct prompts ([8571cff](https://github.com/labscommunity/cascadia/commit/8571cffdec7a1abf277ff2c3807b3ea70c960ce5))
* **dsv4:** sparse-MoE inference engine + distributed pipeline ([c4c9c26](https://github.com/labscommunity/cascadia/commit/c4c9c26ea68a8274766165b54d3597d771c328a1))


### Bug Fixes

* build, CLI and docs bugs found by running every documented command ([c03c68c](https://github.com/labscommunity/cascadia/commit/c03c68c041c159eec3fd89ceaad21fe439733bee))
* **build:** enforce MSRV 1.89, and make the Dockerfile build ([a4aa45a](https://github.com/labscommunity/cascadia/commit/a4aa45a196c369654ecba8f9898ed600f4c06427))
* **cli:** usable errors for models, python and deps ([5173325](https://github.com/labscommunity/cascadia/commit/5173325995b4e133f75feb6d9472cd7c179e627e))
* **dsv4:** apply the finish_reason + truncation fixes to the single-stage path ([9e749d7](https://github.com/labscommunity/cascadia/commit/9e749d75d1f928e13fd61be1c70bd613fbca276f))
* **dsv4:** bound context, fix seeded-sampling parity, and harden the worker ([061c912](https://github.com/labscommunity/cascadia/commit/061c912927f5e3c1345978b5d922ab57010bfbc1))
* **dsv4:** bound the pipeline reply-recv so a dead peer fails fast (no wedge) ([8f197cb](https://github.com/labscommunity/cascadia/commit/8f197cbc8d83a76f69ece760a41d528d3236c93f))
* **dsv4:** exporter carries chat_template.jinja + serving sidecars ([a35340a](https://github.com/labscommunity/cascadia/commit/a35340a8f87328462522fc783a9e4489cced257b))
* **dsv4:** harden streamed prefill against mid-stream failure ([fe56bb6](https://github.com/labscommunity/cascadia/commit/fe56bb638bbbbb891aeb7de5fbfe53f2505f29b3))
* **dsv4:** out-wait a cold slice load on the downstream connect ([ceb7f7f](https://github.com/labscommunity/cascadia/commit/ceb7f7f53b410f5b8abd14650be207cf83023443))
* **dsv4:** reject a manifest whose compress_ratios can't cover its layers ([b3a9480](https://github.com/labscommunity/cascadia/commit/b3a94808ceefdfe357552ca45bfab5adec677b32))
* **dsv4:** reject stage load when manifest exported_layers misses the range ([eb6f760](https://github.com/labscommunity/cascadia/commit/eb6f760df80c96ab6c233e2c32e7db89a18e5298))
* **dsv4:** remove the unusable ov_ir expert export mode ([910ba4c](https://github.com/labscommunity/cascadia/commit/910ba4c54c54b8974f8df67e8d4ad817223e3370))
* **dsv4:** report finish_reason=length when the context window caps decode ([bf637e8](https://github.com/labscommunity/cascadia/commit/bf637e8d1be9627c5665d5496eadde49181aaecf))
* **dsv4:** use one per-token reply deadline (drop the batched-prefill x10) ([ca4df33](https://github.com/labscommunity/cascadia/commit/ca4df3342246d39bd949fde40c402302d250e411))
* **dsv4:** warn instead of silently dropping an over-budget prompt tail ([e3f7427](https://github.com/labscommunity/cascadia/commit/e3f7427ad3c2d81969db84b89b28c8f0bb53fc79))
* **e2e:** find cascadia.exe on Windows ([1189b5b](https://github.com/labscommunity/cascadia/commit/1189b5b06fcc5bb36ef73af053009525737ab82c))
* **engine:** ov-genai requires the tokenizer IRs ([d37db22](https://github.com/labscommunity/cascadia/commit/d37db223ff0b0a87086f54b3ad5fc06def8a48eb))
* pin Intel's key properly, and the bugs review found ([a13d28c](https://github.com/labscommunity/cascadia/commit/a13d28c386c10839cfdccd2ab1fc422f20df9a19))
* review-pass follow-ups ([d0a2a73](https://github.com/labscommunity/cascadia/commit/d0a2a73a10e3ed2a20012b3e0841f8920dbdbd3b))
* **scripts:** install Intel's current GPU drivers, safely ([c5167d7](https://github.com/labscommunity/cascadia/commit/c5167d715792843cae6a6a84bbc18e29cec38f35))
* **transport:** TCP keepalive on inter-rank pipeline sockets ([ad6f5ea](https://github.com/labscommunity/cascadia/commit/ad6f5ea608d1bb895c9c321c18e69f032d834b1f))


### Performance

* **dsv4:** AVX2 batch expert kernel (on-node bit-exact) ([2a2604f](https://github.com/labscommunity/cascadia/commit/2a2604f22b0471f4e9fb378bda0a936e5261e014))
* **dsv4:** AVX2+FMA dot product in GEMV, chunked mmap expert dequant ([d21c0ee](https://github.com/labscommunity/cascadia/commit/d21c0eeeb84917cb8f75c146475ff7e8b0aa210e))
* **dsv4:** batch-union expert kernel for prefill (forward_batch) ([7e17b1d](https://github.com/labscommunity/cascadia/commit/7e17b1dc32ac79de9c3343aea00d6bfc3b5996e3))
* **dsv4:** batch-union MoE in forward_layers_prefill ([49ea601](https://github.com/labscommunity/cascadia/commit/49ea6014bf49aa4947b3427e4e97a4e95bb90f42))
* **dsv4:** batched prefill across the pipeline (ForwardBatchPrefill) ([9ecbf06](https://github.com/labscommunity/cascadia/commit/9ecbf06246cabd7406095167dc66be58783a585d))
* **dsv4:** env-gated per-section decode profiler (DSV4_PROFILE) ([c5dd062](https://github.com/labscommunity/cascadia/commit/c5dd062fb31bb1fe1fed712bd1f4c1fe5d99814e))
* **dsv4:** fused AVX2 int4 dequant-dot for mmap experts ([1520a83](https://github.com/labscommunity/cascadia/commit/1520a83080014bf5cf16c40f942988f58f7a9d77))
* **dsv4:** parallelize GEMV across cores with rayon ([27bae87](https://github.com/labscommunity/cascadia/commit/27bae87e26a4be943b6046d5822361e08b07896e))
* **dsv4:** parallelize the o_proj wo_a GEMV ([4913f43](https://github.com/labscommunity/cascadia/commit/4913f43bba4b5d3d98a3184ebb6a7a0493ddd332))
* **dsv4:** store attention projections as bf16 to halve GEMV bandwidth ([37f01e9](https://github.com/labscommunity/cascadia/commit/37f01e9ff975f07e9b56ac10e175566920f3afb8))
* **dsv4:** stream prefill one-way to pipeline it across ranks ([0375173](https://github.com/labscommunity/cascadia/commit/037517309951fd1361defcb36dec502a0e60edf6))


### Documentation

* add a CLI reference ([2a74a2f](https://github.com/labscommunity/cascadia/commit/2a74a2f3581e1b8a643a0d8db5bb3764161d5cb4))
* **dsv4:** correct the mmap-vs-eager "bitwise" claim ([08d663e](https://github.com/labscommunity/cascadia/commit/08d663e95f45ea4671df65bee0bb6f5748f13a7e))
* **dsv4:** fix ForwardPrefill attribution in the architecture doc ([ed23f4f](https://github.com/labscommunity/cascadia/commit/ed23f4fb2556cc96745d88ba24963da07aa0ec69))
* **dsv4:** trim architecture doc to essentials, document decode perf ([e708d24](https://github.com/labscommunity/cascadia/commit/e708d24e309bde2eb1a4054e8b4b73e203449a0e))
* fix the commands that don't work ([5dfb645](https://github.com/labscommunity/cascadia/commit/5dfb6455b69752a7016c64930655499c7ae0824c))
* install from Intel's unified suite in the by-hand block ([9d29e59](https://github.com/labscommunity/cascadia/commit/9d29e590a1091c755101b3197df030e86d100a1d))


### Testing

* **dsv4:** commit the tiny export fixture tensors so CI can load it ([4b0262e](https://github.com/labscommunity/cascadia/commit/4b0262e6598752603c26b7c10b3d54bc1b9fd85e))
* **dsv4:** skip golden tests when the gitignored fixture is absent ([33dd33d](https://github.com/labscommunity/cascadia/commit/33dd33d7e45f5706728cc2ad14285aee6da14253))


### Miscellaneous

* **deps:** bump spin off yanked 0.9.8 ([de4df2e](https://github.com/labscommunity/cascadia/commit/de4df2e9257b1354c8e6b5d7ed8c03c0a1bec716))

## [0.1.1](https://github.com/labscommunity/cascadia/compare/v0.1.0...v0.1.1) (2026-07-09)


### Features

* [#77](https://github.com/labscommunity/cascadia/issues/77) Part B Path 1 — qwen3_5_moe gate, aliases, single-stage docs ([e01faf4](https://github.com/labscommunity/cascadia/commit/e01faf472791b27d0904f181c3c3dfe353666678))
* **api:** add /v1/completions (OpenAI legacy completions) ([#19](https://github.com/labscommunity/cascadia/issues/19)) ([d194c94](https://github.com/labscommunity/cascadia/commit/d194c944b4f5719e3e3dbcebb018d21b0c113607))
* **api:** add /v1/completions (OpenAI legacy completions) ([#19](https://github.com/labscommunity/cascadia/issues/19)) ([8989b4e](https://github.com/labscommunity/cascadia/commit/8989b4e7adfb240b10c9bc3b85dabfc652acd2bc))
* **api:** expose render_chat_prompt for standalone callers ([63b5477](https://github.com/labscommunity/cascadia/commit/63b5477270cc66c211dc4e3777953ea7830c01fd))
* **api:** non-stream chat emits tool_calls + tool_calls finish_reason ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([e41306f](https://github.com/labscommunity/cascadia/commit/e41306fe4c6fa380e60324788e900d240396b3ad))
* **api:** Ollama dialect + tool-call plumbing ([9fe99ed](https://github.com/labscommunity/cascadia/commit/9fe99ed3ef092977dc30ccd260a699504153fef2))
* **api:** OpenAI sampling params + finish_reason + streaming usage ([#14](https://github.com/labscommunity/cascadia/issues/14)) ([58143e6](https://github.com/labscommunity/cascadia/commit/58143e6edeae7c7df4072f2a66b68ca527df582e))
* **api:** OpenAI sampling params, finish_reason, streaming usage ([#14](https://github.com/labscommunity/cascadia/issues/14)) ([21c15f2](https://github.com/labscommunity/cascadia/commit/21c15f2e419ae787e055a442a09d80a49b65ff32))
* **api:** OpenAI tool calling (function calling) [cascadia-enterprise[#44](https://github.com/labscommunity/cascadia/issues/44)] ([780cbf4](https://github.com/labscommunity/cascadia/commit/780cbf43ea9a90853066eb1a7026e5855da9d2d1))
* **api:** parse_tool_calls for Llama + Qwen tool output ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([79ad3bc](https://github.com/labscommunity/cascadia/commit/79ad3bcbf38226fe4c85939ec01b14d0b3fad564))
* **api:** render chat_template from tokenizer_config.json (Jinja2) ([339091a](https://github.com/labscommunity/cascadia/commit/339091a891ee839d02b94198002e0da6e6c3b448))
* **api:** renderer forwards tools + message tool fields ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([150748e](https://github.com/labscommunity/cascadia/commit/150748e5e963969fa8dc52695f58cf6c0f7e9cc2))
* **api:** SSE streaming, logprobs, cancellation, introspection, tracing ([d1e858a](https://github.com/labscommunity/cascadia/commit/d1e858a2a79bf760882f394a121a5f7b5fbc80a2))
* **api:** streaming emits single indexed tool_calls delta ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([ee140c0](https://github.com/labscommunity/cascadia/commit/ee140c0b580cf82bd52d0685ebc66aeca48f7405))
* **api:** tool-calling request/response schema types ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([963c5c2](https://github.com/labscommunity/cascadia/commit/963c5c241dc5a3a385cd6fc7c41f212e01b90187))
* **cascadia-api:** expose render_chat_prompt for in-process embedders ([0382ec3](https://github.com/labscommunity/cascadia/commit/0382ec3d44a4dab0270c033ebef0ab9062aaf60d))
* **cascadia-api:** expose render_chat_prompt for in-process embedders ([db2a3a0](https://github.com/labscommunity/cascadia/commit/db2a3a044cfe1e2079d85f0a6668ed6c78585e8f))
* chunk-level n_tokens for accurate tok/s with spec-decode ([7c02e55](https://github.com/labscommunity/cascadia/commit/7c02e55c15e28c3e8644782a69d4a0298da0b198))
* **cli:** add --version flag ([4dd173e](https://github.com/labscommunity/cascadia/commit/4dd173ec43853713219b802a0662c7cf009eeea2))
* **cli:** add doctor, run, discover, and completions subcommands ([acce033](https://github.com/labscommunity/cascadia/commit/acce033510402b85e64da0eb373b549b394bc703))
* **cli:** plumb OpenVINO performance properties through all engines ([525757f](https://github.com/labscommunity/cascadia/commit/525757f2d73d851357de309ea11138f145ea66e3)), closes [#13](https://github.com/labscommunity/cascadia/issues/13)
* **cli:** plumb OpenVINO performance properties through the CLI ([#13](https://github.com/labscommunity/cascadia/issues/13)) ([f10cf40](https://github.com/labscommunity/cascadia/commit/f10cf401511003658b0bbb838b33fb5b0e7b0442))
* **cli:** profile-devices subcommand — per-device tok/s for [#41](https://github.com/labscommunity/cascadia/issues/41) step 1 ([#45](https://github.com/labscommunity/cascadia/issues/45)) ([8a20154](https://github.com/labscommunity/cascadia/commit/8a20154e6ec212c8b1a4555d207ef2031ee28e3f))
* **cli:** type OV mode flags as ValueEnum; warn on ignored perf flags ([db171f1](https://github.com/labscommunity/cascadia/commit/db171f11aebada6c0bb9300d12997eca89a803a6))
* **cluster:** mDNS discovery + master election + automatic placement ([70cd54b](https://github.com/labscommunity/cascadia/commit/70cd54b7ece6b3c12ebb9ff4ebe949c94188062c))
* **dashboard:** chat playground with streaming SSE + decode stat bar ([ab266f1](https://github.com/labscommunity/cascadia/commit/ab266f101941b02ba1ef68d39dc8ce57e59cf462))
* **dashboard:** cluster web UI + chat playground (tahoma-dashboard crate) ([73444fd](https://github.com/labscommunity/cascadia/commit/73444fd4f260fb6105a39e6048cfb71b75298fa8))
* **dashboard:** ClusterStrip on the chat surface ([191f5e7](https://github.com/labscommunity/cascadia/commit/191f5e71521afef82654c26f203e35615f0c8e58))
* **dashboard:** live request/token stats + per-node system specs ([2994caf](https://github.com/labscommunity/cascadia/commit/2994caf3db0564cb5425695e8f7587be3b133951))
* **dashboard:** node cards + latency matrix on the Cluster page ([88fcec7](https://github.com/labscommunity/cascadia/commit/88fcec71361303e83313ea388e75b257e6cae8aa))
* **dashboard:** scaffold tahoma-dashboard crate with /api/topology + /api/stats ([560a83f](https://github.com/labscommunity/cascadia/commit/560a83f0fa7c3f50d389ec538142d8c0710a351c))
* **dashboard:** scaffold Vite + React + Tailwind SPA with cascadia design tokens ([17534a5](https://github.com/labscommunity/cascadia/commit/17534a5b9bd601cd0c1a4b8c140a7fbce876b67f))
* **dashboard:** serve dashboard alongside API in tahoma worker --api ([660e13e](https://github.com/labscommunity/cascadia/commit/660e13e87f0a2c5ff9f8fcdce0c5352a598081b3))
* **dashboard:** TCP latency probe loop populates the matrix ([871c09e](https://github.com/labscommunity/cascadia/commit/871c09eef4f358011a5d3de0cbd1f17dc65bfef3))
* **dashboard:** use the Cascadia brand logo + favicon ([70335e6](https://github.com/labscommunity/cascadia/commit/70335e696be96a92309002d7c6e498fc621fb869))
* **dist_spec:** per-round streaming (was buffering all output to one chunk) ([bf0d68e](https://github.com/labscommunity/cascadia/commit/bf0d68eba0b06a71bdf0e9498fb151151d8d71f9))
* **download:** HuggingFace model registry + pull endpoints ([00f6019](https://github.com/labscommunity/cascadia/commit/00f60199ce74a4d40d4a620d0d859f66d056ac67))
* **engine,transport,runner:** pipeline-link reliability & dead-peer recovery ([d10e44a](https://github.com/labscommunity/cascadia/commit/d10e44a12f2970c265a4a10d09a2fa7e75e621c1))
* **engine:** Engine::step returns EngineResult so failures are visible ([826a01b](https://github.com/labscommunity/cascadia/commit/826a01b0f43a1901abdba67e9f44a0449b3650a1))
* **engine:** Gemma 4 runtime (--engine gemma4) — single + multi-stage pipeline-parallel ([3ff7809](https://github.com/labscommunity/cascadia/commit/3ff7809630e59e22a0f757541f7034cdbe6d82ac))
* **engine:** implement engine-side cancel() propagation ([#15](https://github.com/labscommunity/cascadia/issues/15)) ([4634c74](https://github.com/labscommunity/cascadia/commit/4634c749631bb4f78fd4f3a6938cbf8f505dedcb))
* **engine:** implement engine-side cancel() propagation ([#15](https://github.com/labscommunity/cascadia/issues/15)) ([e6a5ce2](https://github.com/labscommunity/cascadia/commit/e6a5ce2479a336bb4a5006f950a5c1f2a2c60fd0))
* **engine:** KV cache via forward_layers_cached ([ccec98c](https://github.com/labscommunity/cascadia/commit/ccec98ce57d8f283986b8e3be54fb84d14442d67))
* **engine:** real cancel on ov-runtime + mock; document trait contract ([cc430ea](https://github.com/labscommunity/cascadia/commit/cc430eaab9f41a3c289cbc9a1bb6ec2ad55f0f00))
* **engines:** add ov-genai single-stage LLMPipeline engine ([6a22922](https://github.com/labscommunity/cascadia/commit/6a22922e22f5446f2e5e337feef633f83ce04f53))
* **engines:** add ov-genai single-stage LLMPipeline engine ([0c95e1a](https://github.com/labscommunity/cascadia/commit/0c95e1a3a58285a0f63a4d890d6ed9c42e062d93))
* **engine:** single-stage OpenVINO Runtime engine via optimum-intel ([8d7551c](https://github.com/labscommunity/cascadia/commit/8d7551cde156a88e5e7c944389ad49051b99807d))
* **engines:** plumb OV plugin properties through ov-runtime + ov-dist-spec ([ccd2aa8](https://github.com/labscommunity/cascadia/commit/ccd2aa83108d8c68bee15787f929344eeb5de340))
* **export:** --free-source-shards for in-place re-quantization ([371fbfc](https://github.com/labscommunity/cascadia/commit/371fbfc2c78c61c6593bd4b859fe4c0cd0d4e3f2))
* **export:** MiniMax-M2 exporter to OV-IR sparse-MoE layout ([bb1f222](https://github.com/labscommunity/cascadia/commit/bb1f2226521ef5be0a095c154465feb911fa6c4d))
* **export:** NF4 (distribution-matched 4-bit) expert quant ([e70f656](https://github.com/labscommunity/cascadia/commit/e70f656efc6f9a0923e3f0274016a54dbddf96dd))
* **export:** per-component precision flags (--shell-quant / --head-quant) ([dfda2c1](https://github.com/labscommunity/cascadia/commit/dfda2c1c34ef0867b333eb2a0c37d6569fedcb3b))
* **export:** streaming per-layer full-model export for MiniMax-M2 ([8046c2f](https://github.com/labscommunity/cascadia/commit/8046c2f1b7bea0b54b3ea57e088fd58710d27d49))
* **gemma4:** forward cross-stage shared KV for multi-stage E2B/E4B ([c4969d8](https://github.com/labscommunity/cascadia/commit/c4969d808ed0e4af085f7c035e1a876b17f324fd))
* **gemma4:** IR-surgery tool — text-only shards from OpenVINO VLM IR ([899d354](https://github.com/labscommunity/cascadia/commit/899d354c2416d3a442c709e383cfc19ef998e0ca))
* M3' decode prototype — 64/64 greedy parity over 2-stage chain ([59a4e1f](https://github.com/labscommunity/cascadia/commit/59a4e1feeac2b75f8f8a61b170ecf0ce482d09ef))
* MiniMax-M2 support (single-stage OV-IR sparse-MoE) ([e94ca21](https://github.com/labscommunity/cascadia/commit/e94ca21daa9181bc2b802a601a75c73f10e7293d))
* MVP runtime — OpenVINOEngine, Runner, OpenAI API, CLI ([dbbc252](https://github.com/labscommunity/cascadia/commit/dbbc252926bb7b7a8369f106cfe617fde0132fe7))
* **ov-dist-spec:** distributed speculative decoding engine ([bb8f3eb](https://github.com/labscommunity/cascadia/commit/bb8f3eb569a3f7de8d2eefe7221c0c61af7bbde6))
* **ov-dist-spec:** v5 shards with mask-based KV rewind ([fbc19c9](https://github.com/labscommunity/cascadia/commit/fbc19c98efd719693b1dd1df38b0cb6383b2ae07))
* **ov-optimum:** speculative decoding via assistant_model ([826485d](https://github.com/labscommunity/cascadia/commit/826485db2140e977d43eeeea3b8bf1213c014bd1))
* **ov-spec:** manual mask-based-rewind speculative decoding ([cd53c53](https://github.com/labscommunity/cascadia/commit/cd53c5330ccc631e7cd8977a9427d54bdddc69b7))
* **ov-spec:** per-token streaming via generator + incremental decode ([2594860](https://github.com/labscommunity/cascadia/commit/259486052d7bc5bed5830bebae932abe33cbc61a))
* **ov:** auto-export OV IR from HuggingFace model id ([9122992](https://github.com/labscommunity/cascadia/commit/9122992147c7bd3e573ad9d126b71f400370e910))
* **ov:** multi-stage OV Runtime engine with stateful KV cache ([0076ba9](https://github.com/labscommunity/cascadia/commit/0076ba9259fe59fd9a6ae4e9b23afce05a6bb546))
* **ov:** per-token streaming chunks via TextIteratorStreamer ([4b342d8](https://github.com/labscommunity/cascadia/commit/4b342d824b905b7df97f32af02dd2508733ed63d))
* **parallel:** pytorch-tp engine — column/row split + ring all-reduce ([dbb5a69](https://github.com/labscommunity/cascadia/commit/dbb5a69d96bc55e6909110536eaf25fb4be4348d))
* **parallel:** tensor-parallel foundation — ring all-reduce + ShardSpec ([e97c741](https://github.com/labscommunity/cascadia/commit/e97c741c2779f787c3cba43a4a4e2fa635fad185))
* **placement:** `cascadia run-placement` — launch the heterogeneous pipeline ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([41662b2](https://github.com/labscommunity/cascadia/commit/41662b2f2c7c94c343984c803c6d9afc557ae567))
* **placement:** exact memory-capped ILP solver + `cascadia place` ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([6f207cb](https://github.com/labscommunity/cascadia/commit/6f207cb409ebd3b12dcb15274d47060841c79363))
* **placement:** model the shared UMA pool as a global memory gate ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([156e73c](https://github.com/labscommunity/cascadia/commit/156e73c60989a1af971e5a819a6533135576d138))
* **placement:** three-tier {iGPU, NPU, CPU} ILP placement ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([1799217](https://github.com/labscommunity/cascadia/commit/17992176c0d61bf6ac7df2d394669eed13b932dc))
* **profile:** fingerprint-keyed profile cache — reuse a matching profile, skip re-measure ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([d75225f](https://github.com/labscommunity/cascadia/commit/d75225f5b93f3c6589bf34a57a3cd5c16d865969))
* **profile:** per-stage per-device cost profiler — `cascadia profile-stages` ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([ce82afe](https://github.com/labscommunity/cascadia/commit/ce82afea0cf283c14a75467269851ff3558bdd48))
* qwen36 full-attention layer cut bit-exact (M2' brick 3 complete) ([ecb9cad](https://github.com/labscommunity/cascadia/commit/ecb9cad568c80c8a13a08d35a175817f5423f860))
* qwen36 IR-surgery probes — one-expert slice parity proven ([e9a2f45](https://github.com/labscommunity/cascadia/commit/e9a2f45843e399dd80865be0bba5b870fed441ff))
* qwen36 MoE semantic parity proven (M2' brick 2) ([e703c82](https://github.com/labscommunity/cascadia/commit/e703c82a9d61cf3067746e5dc2d079de501974e8))
* qwen36 shard exporter working — 2-stage chain validated (M2') ([070a5a1](https://github.com/labscommunity/cascadia/commit/070a5a1694b7571feb3c79b29553957e01e9934a))
* qwen36 shell extraction proven bit-exact (M2' brick 3) ([6f081f0](https://github.com/labscommunity/cascadia/commit/6f081f003061e8b15ce2bfd25e56ca6c4a6fba64))
* qwen36-moe staged engine (M3' v1) ([ad6f998](https://github.com/labscommunity/cascadia/commit/ad6f998dc73396b61d8edf133b5aaccfdc201975))
* run cascadia shards on the Intel NPU — single + pipeline-parallel multi-stage ([#37](https://github.com/labscommunity/cascadia/issues/37)) ([74c59ab](https://github.com/labscommunity/cascadia/commit/74c59ab303df736f14d616840f6f08553e0a3678))
* **runner:** concurrent generate() — share engine via lock + per-task buffers ([db054e6](https://github.com/labscommunity/cascadia/commit/db054e690276b5d3437680fba8412a338da65e05))
* **runtime:** pipeline-parallel static-KV decode across NPU stages ([#37](https://github.com/labscommunity/cascadia/issues/37)) ([4681931](https://github.com/labscommunity/cascadia/commit/468193107f106fda7be6fcaf76a24f8c876ea0b0))
* **runtime:** stateless static-KV decode path for NPU shards ([#37](https://github.com/labscommunity/cascadia/issues/37)) ([5bddf71](https://github.com/labscommunity/cascadia/commit/5bddf710c5bdf67c44ad4f43234520dedce1142a))
* **rust:** hard rewrite — Rust workspace replaces Python tree, all engines validated ([09e732b](https://github.com/labscommunity/cascadia/commit/09e732b7997faa1e17ac9b630dbfbb3494a9f33b))
* **rust:** ov-runtime distributed e2e validated on v3 shards (alpha+charlie) ([f2735a2](https://github.com/labscommunity/cascadia/commit/f2735a25a546f6d9fe467e9aadc4c2f67a1c4515))
* **rust:** port ov-runtime + ov-dist-spec engines + extend OV FFI shim ([b82149b](https://github.com/labscommunity/cascadia/commit/b82149b9603188b7fbd0d13c4b1dce7d7292f95d))
* **rust:** scaffold Rust port — workspace + foundation crates + ov-genai engine ([7f259f0](https://github.com/labscommunity/cascadia/commit/7f259f0f2be2287937f3a8481b7bfa117d4a1200))
* **rust:** tahoma-discovery (mDNS) + tahoma-download (registry+HF) + STATUS ([a0ec75f](https://github.com/labscommunity/cascadia/commit/a0ec75f6fab8d28da7f12143608d2997989df9e0))
* scaffold engine plugin layer ([c0754ce](https://github.com/labscommunity/cascadia/commit/c0754cee8a1b784d925f150504644a6594472e6c))
* **shard:** broaden exporter architecture support — Phi-3, Gemma-2, .bin loading, MoE rejection ([#58](https://github.com/labscommunity/cascadia/issues/58)/[#59](https://github.com/labscommunity/cascadia/issues/59)/[#60](https://github.com/labscommunity/cascadia/issues/60)/[#61](https://github.com/labscommunity/cascadia/issues/61)) ([62d8b48](https://github.com/labscommunity/cascadia/commit/62d8b4809f90ae611acb0b4dc8a2d531011f9e84))
* **shard:** broaden exporter architecture support — Phi-3, Gemma-2, .bin, MoE-reject ([0faa2e7](https://github.com/labscommunity/cascadia/commit/0faa2e7bac5d06f4b0ec3d1a4a77564be3eeb52a))
* **shard:** dispatch gemma-4 OpenVINO-IR input to the text-surgery exporter ([70e021b](https://github.com/labscommunity/cascadia/commit/70e021b33e1330cd6a6c6efc08fa4cf01888b70f))
* **shard:** forward NPU static-export flags through `cascadia shard` ([14bce43](https://github.com/labscommunity/cascadia/commit/14bce43968a6270bfc8e1c900785403757d7dbf5))
* **shard:** Gemma 4 exporter (dedicated export_gemma4.py) + dispatch ([1170893](https://github.com/labscommunity/cascadia/commit/1170893d1da1dbff2ce915a538acf2dd2aa5473e))
* **shard:** Gemma 4 exporter (port from rainier prototype) ([a306e4b](https://github.com/labscommunity/cascadia/commit/a306e4b81deefd97b09623204ed22125b9d98692))
* **shard:** NPU static-export flags + gemma-4 IR-surgery (text-only shards) ([6dbbe5b](https://github.com/labscommunity/cascadia/commit/6dbbe5be2e1844dd3f30d3a78e7bf852eccb7eca))
* **shard:** NPU-targeted stateless + static-shape export mode (toward [#37](https://github.com/labscommunity/cascadia/issues/37)) ([f7e104f](https://github.com/labscommunity/cascadia/commit/f7e104f770dfddf03dc215d9494e58021d90feef))
* **shard:** partial rotary support + config-first arch rejection (reconcile [#47](https://github.com/labscommunity/cascadia/issues/47)) ([014221a](https://github.com/labscommunity/cascadia/commit/014221aa9a2f265b1c0c7108fe16d5496449dc3b))
* **shard:** R1-Distill alias registry + docs/architectures/r1-distill.md ([22ecef5](https://github.com/labscommunity/cascadia/commit/22ecef54f8f25df393c320ad5c319028aa60d67f))
* **shard:** R1-Distill alias registry + per-family deep-dive doc ([ac4c071](https://github.com/labscommunity/cascadia/commit/ac4c071fada74eee581c980e39ba22cc5bf41f03))
* **shard:** standalone model-sharding via 'tahoma shard' ([b36c9dc](https://github.com/labscommunity/cascadia/commit/b36c9dc77b9aa38557fdfa6fde9180965dbce570))
* **shard:** standalone model-sharding via 'tahoma shard' ([0b29362](https://github.com/labscommunity/cascadia/commit/0b293623e2f32a89e7d2b4dbcc3b4fd753a13e65))
* **shard:** support partial rotary + reject Gemma 3/4, gpt-oss, Mamba config-first ([fa64a5d](https://github.com/labscommunity/cascadia/commit/fa64a5d5646011b21793676ab935fa5336dc9d33))
* **shard:** support Qwen3 (q_norm/k_norm + decoupled head_dim) ([87a0e03](https://github.com/labscommunity/cascadia/commit/87a0e03f47979f13cb358f7a15a9182e6efcc3b7))
* **shim:** expose input rank/shape/dtype getters on Runtime ([85d2bfa](https://github.com/labscommunity/cascadia/commit/85d2bfa2def98f37053a3a6a3993d7b52e194a41))
* **sparse-moe:** asymmetric M2 layer split + int4_bin-&gt;ov_ir expert converter ([9f7626d](https://github.com/labscommunity/cascadia/commit/9f7626db713f12dce701b106a58583cd0b0ca1ed))
* **sparse-moe:** iGPU router-split for MiniMax-M2 shells ([028a11b](https://github.com/labscommunity/cascadia/commit/028a11b91ce0af977e00d871c04c3475269a3bb9))
* **sparse-moe:** Kimi K2.6 Rust engine — sparse top-8 dispatch + Rust shells + int4 GEMM ([#7](https://github.com/labscommunity/cascadia/issues/7)) ([aedee33](https://github.com/labscommunity/cascadia/commit/aedee33e8cbb764056fcd71b5cb18f905d9034ad))
* **sparse-moe:** layer-0 KV cache + pre-alloc KV + dispatch lift + multi-stage sampling ([#10](https://github.com/labscommunity/cascadia/issues/10)) ([208104e](https://github.com/labscommunity/cascadia/commit/208104e8e26463eb376cc77a9fc5a09cb8df0e56))
* **sparse-moe:** OV-IR shell backend for MiniMax-M2 ([fd84a34](https://github.com/labscommunity/cascadia/commit/fd84a344cb491488542276cd9719cc21a08a7a20))
* **sparse-moe:** pipeline-parallel inference + Rust shells ([#9](https://github.com/labscommunity/cascadia/issues/9)) ([8874c93](https://github.com/labscommunity/cascadia/commit/8874c93a067dd2f836e612abe0d3f2f2b2641017))
* **sparse-moe:** pipeline-parallel MiniMax-M2 across ranks ([9f70298](https://github.com/labscommunity/cascadia/commit/9f702982259b6df7b0d35f87f1b5a48440367190))
* **sparse-moe:** pipeline-parallel MiniMax-M2 across ranks ([592a5d9](https://github.com/labscommunity/cascadia/commit/592a5d908d74dcb2121b4cebd955a889555ab815))
* **sparse-moe:** repetition-penalty sampling + int4_bin expert backend ([832d6a8](https://github.com/labscommunity/cascadia/commit/832d6a8daf4070fab5ff21882a763af97a2c2a34))
* **sparse-moe:** run MiniMax-M2 on Intel iGPUs via router-split ([8431e59](https://github.com/labscommunity/cascadia/commit/8431e592d4969c5eb2550911fbe3fe40b89cecb0))
* **sparse-moe:** static prompt KV-prefix cache (single-stage, opt-in) ([8777680](https://github.com/labscommunity/cascadia/commit/8777680e5c399129ea2a7dd1a156f6146b3e2a12))
* stable public API exports + drop unused import ([7e3669d](https://github.com/labscommunity/cascadia/commit/7e3669d6990b08822e7b3dd3696ebe764398fa71))
* **tool-calling:** parse Qwen3 &lt;function=…&gt;&lt;parameter=…&gt; XML tool-call dialect ([7266dc1](https://github.com/labscommunity/cascadia/commit/7266dc148ffcabc0d002a123b9f478dd12dc734d))
* **transport:** config-settable activation recv timeout ([09ac95c](https://github.com/labscommunity/cascadia/commit/09ac95c39fbea679d9c51120182d9586bfd8357b))
* **transport:** env-configurable activation recv timeout ([45f210e](https://github.com/labscommunity/cascadia/commit/45f210e8471b8ea3705740214d87b10225c8fe01))
* **transport:** env-configurable activation recv timeout ([3ead587](https://github.com/labscommunity/cascadia/commit/3ead587e570e3372c99a75a986328737eb932cd8))
* **transport:** frame-start idle ceiling for black-holed peers ([62134da](https://github.com/labscommunity/cascadia/commit/62134dacfd79a3232f44dafbe7ad4ee4f5470be9))
* **transport:** loud, periodic feedback while waiting for a peer ([d8039f4](https://github.com/labscommunity/cascadia/commit/d8039f41ff56ec59916781b2a1393c23d13ea846))
* VLMPipeline support in shim + ov-genai engine ([#77](https://github.com/labscommunity/cascadia/issues/77) Path 1) ([04729ac](https://github.com/labscommunity/cascadia/commit/04729ac5ca5dcce1dc6337aa31a8a9667b98bc26))


### Bug Fixes

* **api,topology,discovery,cli:** address review — counters, specs, discovery ([d200b6e](https://github.com/labscommunity/cascadia/commit/d200b6e64819c0d0aecb576552ef8adef32f620b))
* **api:** bare-JSON tool-call requires args/params (M3 partial) ([ab1ae13](https://github.com/labscommunity/cascadia/commit/ab1ae13f7f78a172f693673860dac8f7c9d3895c))
* **api:** brace-balanced tool_call JSON scan (M2) ([8399d1a](https://github.com/labscommunity/cascadia/commit/8399d1a84c634d27b49848ac799364e10cf0f82e))
* **api:** build_choice requires tools_present to parse ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([a374e30](https://github.com/labscommunity/cascadia/commit/a374e30cb8be49beda1e635a5f55eda1b4b2e10f))
* **api:** legacy chat template renders only the latest user turn ([01c42dd](https://github.com/labscommunity/cascadia/commit/01c42dd616c5528bd76e93d050b0b9f2a214d265))
* **api:** load chat_template.jinja sibling + enable minijinja macros ([90b3c79](https://github.com/labscommunity/cascadia/commit/90b3c794a338e559c57b91515da2a445354d3dd2))
* **api:** per-token SSE flush — three layers were batching ([ecfeec9](https://github.com/labscommunity/cascadia/commit/ecfeec9a97e463aca10e24543ef6e4743c181fe3))
* **api:** reject empty chat_template.jinja + parse template once at startup ([2f9d477](https://github.com/labscommunity/cascadia/commit/2f9d4773cad326f46d7333d4aafc0035771e9995))
* **api:** reject empty prompt with 400 instead of an empty 200 ([e91d12f](https://github.com/labscommunity/cascadia/commit/e91d12f2c37a564d26f2e51376b033b8c499393e))
* **api:** render tool_call arguments as object for HF chat templates ([89b0b44](https://github.com/labscommunity/cascadia/commit/89b0b44205a5935d7496605247c37922bc0bfa3a))
* **build:** actionable error when INTEL_OPENVINO_DIR is missing or wrong ([47fa446](https://github.com/labscommunity/cascadia/commit/47fa446220bfc6eb3e6196c89cd8ee7ca493a8d4))
* cargo fmt + accurate TargetSendHandle drop-semantics doc ([3a9393d](https://github.com/labscommunity/cascadia/commit/3a9393d8cb8518d95974c00f0af688c0e74d3b3f))
* **cascadia-api:** render chat prompts via ChatPromptRenderer ([16bbd96](https://github.com/labscommunity/cascadia/commit/16bbd966b500d11520918c3a598b8ec7cad6bc53))
* **cli:** gate NPU-only OV properties to the ov-genai engine ([3bc8f5d](https://github.com/labscommunity/cascadia/commit/3bc8f5d1a5948123e6aff662c1709e71ff04fa6d))
* **cli:** SIGTERM/SIGINT-graceful worker shutdown ([2bb047b](https://github.com/labscommunity/cascadia/commit/2bb047b19f2189b2b8f1846fad199acfb58518aa))
* **dashboard:** bind a probe listener so mock-engine workers are reachable ([76985a1](https://github.com/labscommunity/cascadia/commit/76985a137026ec193c07ec862d210f385b24ae49))
* **dashboard:** every worker advertises via mDNS, not just rank 0 ([654e0fe](https://github.com/labscommunity/cascadia/commit/654e0fe9c7ae7547f3a969c639be4bb11fa17701))
* **dashboard:** live indicator now fires — self-heartbeat + 60 s freshness window ([d94f79c](https://github.com/labscommunity/cascadia/commit/d94f79cc0ddab6462286b192f26e780e0afa21c6))
* **dashboard:** mDNS presence is the liveness signal, not last_seen ([ce10488](https://github.com/labscommunity/cascadia/commit/ce10488bf8c19b62d4f63acd02ef34260e3dd642))
* **dashboard:** ModelPicker always renders a select ([c32dd6c](https://github.com/labscommunity/cascadia/commit/c32dd6cb4017dda6b05014f7102680c66fae642d))
* **dashboard:** SPA routing + chat error handling + UI cleanup ([8b567d4](https://github.com/labscommunity/cascadia/commit/8b567d4e384198b636b214fbd15a9ac640ca4d16))
* **dashboard:** use the circle mark as the favicon ([daac2fc](https://github.com/labscommunity/cascadia/commit/daac2fc24007b168e3e5e61e5d2603a359b74231))
* **deploy:** move systemd StartLimit* to [Unit] so the restart cap applies ([eb1f85b](https://github.com/labscommunity/cascadia/commit/eb1f85b9d2d0a9cfc2f972011d301ecdb95d2c25))
* **dist_spec:** apply multi-EOS + post-loop truncate to spec-decode ([875c181](https://github.com/labscommunity/cascadia/commit/875c1815017dc00794026b001142197580bf7c08))
* **engine-mock:** Ok-wrap the __engine_error__ sentinel return ([c86cd0d](https://github.com/labscommunity/cascadia/commit/c86cd0d73ec73f0b3132f353284399087ad80606))
* **engine-openvino:** migrate Qwen36Engine::step to EngineResult ([2f6b213](https://github.com/labscommunity/cascadia/commit/2f6b213a148ba559b91dd0ee5d22ce83772feb29))
* **engine-openvino:** qwen36 step propagates relay Err + parity test compiles ([469da4c](https://github.com/labscommunity/cascadia/commit/469da4c0ade9b4742a23b45c4bc8d9239c430d01))
* **engine,runner:** route step() Err to the failed task's stream ([c8048ed](https://github.com/labscommunity/cascadia/commit/c8048ed3b60faf5532a183d1f88a6b30223e71ef))
* **engine:** 1-token prompt prefill keeps the strict reply deadline ([e9d4cc7](https://github.com/labscommunity/cascadia/commit/e9d4cc77849bf68522b4dcb8ecdaba8e98476890))
* **engine:** back off on step error; saturate prefill timeout multiply ([ae458b1](https://github.com/labscommunity/cascadia/commit/ae458b1b2d6e32eb714de7e0bb5b1b9dc2a51138))
* **engine:** drop cancel() dupes main's [#15](https://github.com/labscommunity/cascadia/issues/15) already provides ([3a72f5d](https://github.com/labscommunity/cascadia/commit/3a72f5dd1c41730e1522711bc0f17677268e88ca))
* **engine:** finish EngineResult migration against main's newer code ([e26fab4](https://github.com/labscommunity/cascadia/commit/e26fab47b2c5df7bac158e6b8fb2d312a9936e7d))
* **engine:** handle 3D-padded token return from wire format ([7769e5a](https://github.com/labscommunity/cascadia/commit/7769e5a43199623604b70b27483ad447a5d4c37f))
* **engine:** is_connection_fatal covers EngineError::Io; dist-spec delegates ([89a3f32](https://github.com/labscommunity/cascadia/commit/89a3f3220e04b6440881308496100da9dace4094))
* **export:** correct partial-rotary derivation + resumable full export ([5f5186f](https://github.com/labscommunity/cascadia/commit/5f5186ff2216baca669b4c2b4f3b775d4f85c355))
* **gemma4-export:** asymmetric global KV heads + k_eq_v for 31B ([1593d94](https://github.com/labscommunity/cascadia/commit/1593d94ba9665d59e6101a1aeb654d7a705da4a3))
* **gemma4-export:** asymmetric global KV heads + k_eq_v for Gemma-4-31B ([abd0808](https://github.com/labscommunity/cascadia/commit/abd08086f70ed6b52bcfaabe46d9b9abc48acf01))
* **gemma4-export:** guard present-but-None/0 global KV-head field ([91ff356](https://github.com/labscommunity/cascadia/commit/91ff356055c194270a883cb021db555d27f10a32))
* **gemma4-export:** init-bearing stateful KV so reset_state yields a batch-1 {1,h,0,d} state (CPU plugin + runtime) ([6ce8cfa](https://github.com/labscommunity/cascadia/commit/6ce8cfae691ff8520afaf85484097d2ecd2e52d8))
* **gemma4:** actually remove the _grafted temp dir on Windows ([5cb7ef6](https://github.com/labscommunity/cascadia/commit/5cb7ef6c2777d6aace71c0bf891614f5b9a75499))
* **gemma4:** emit rainier-v3 stage layout so ov-runtime loads the N&gt;1 slices ([be24567](https://github.com/labscommunity/cascadia/commit/be24567ce5fc027889d6f091bda347f9353b8538))
* **gemma4:** fail on missing tokenizer.json, warn loudly on absent chat template ([04be8a3](https://github.com/labscommunity/cascadia/commit/04be8a3e6cd9179303bb25e9daeb251500e73752))
* **gemma4:** make pipeline_config.json a true completion marker ([4bccd1f](https://github.com/labscommunity/cascadia/commit/4bccd1f8c62476b622d440cd1f497309d3aceb2c))
* **gemma4:** mid-stage mask seq-dim from hidden_states + ship chat_template ([c94c594](https://github.com/labscommunity/cascadia/commit/c94c5946c73737ff1951e2a8c1641f8fac36f238))
* **gemma4:** neutralize token_type_ids + coerce transformers-5 tokenizer_config ([f0be765](https://github.com/labscommunity/cascadia/commit/f0be765dff3054ffcc7530d77a639f2f34aee522))
* **gemma4:** only tolerate missing deps in the tokenizer-BOS regen ([954dc42](https://github.com/labscommunity/cascadia/commit/954dc420c83c5ee0a524783e120c0fec9da909da))
* **gemma4:** relay hidden_states as f32, not f16 ([07ba3e5](https://github.com/labscommunity/cascadia/commit/07ba3e589439b9b18ab1f297b2cfbb79e020e46f))
* **gemma4:** restore cross-KV frame-count guard + review nits ([72725dc](https://github.com/labscommunity/cascadia/commit/72725dcf1c7cf00cac827babafd9437908a70eb1))
* **gemma4:** revert KV-sink hard-fail — broke 26B heterogeneous KV ([b9d34ce](https://github.com/labscommunity/cascadia/commit/b9d34ceed88e4845f34f0817a4a3dca623272059))
* **gemma4:** review fixes — half-open layer_end, KV-sink hard-fail, guards ([3c95664](https://github.com/labscommunity/cascadia/commit/3c95664516f658e2f497d5e855ec74c97b9a2583))
* **gemma4:** slice sink-ownership by global-layer scope + shape-aware KV rewire ([59936e8](https://github.com/labscommunity/cascadia/commit/59936e8febba979532eeafb6a8c9edcb29e15196))
* **gemma4:** surface silently-dropped sinks and the --stage safety-net bypass ([a136ecc](https://github.com/labscommunity/cascadia/commit/a136ecccd02d29f9dfa657272af1ab167d198ffa))
* **gemma4:** tokenizer/ subdir + hidden_states f16 inter-stage input for ov-runtime ([7e3b6a4](https://github.com/labscommunity/cascadia/commit/7e3b6a4b201275c0f5b3022445367e4998efec00))
* **loader:** pass position_embeddings in forward_layers_cached ([b826126](https://github.com/labscommunity/cascadia/commit/b8261261ab9581acf91c46384a8ef764575e855d))
* **openvino:** don't close warn streak on idle first-stage Ok ([f2fcd1d](https://github.com/labscommunity/cascadia/commit/f2fcd1d7858abdd52af025f80e9f448901e9a382))
* **openvino:** rate-limit gemma4 step() WARN via shared limiter ([#30](https://github.com/labscommunity/cascadia/issues/30)) ([c24d83a](https://github.com/labscommunity/cascadia/commit/c24d83a864d57b82a2bcdd3a0a90c5e203d5bc13))
* **openvino:** rate-limit per-call WARN on failing step() ([6d53e91](https://github.com/labscommunity/cascadia/commit/6d53e9137eaf7c9f5b05784df916d4f0397a87c3))
* **openvino:** rate-limit per-call WARN on failing step() ([0bc9616](https://github.com/labscommunity/cascadia/commit/0bc961622969120e0ebf71b5926a8c646b07a8cf))
* **ov-genai:** populate /v1/chat/completions usage token counts ([#55](https://github.com/labscommunity/cascadia/issues/55)) ([693c4fd](https://github.com/labscommunity/cascadia/commit/693c4fd3b1b9ac148f47eedf6c9c2389e4e6ee97))
* **ov-genai:** populate /v1/chat/completions usage token counts ([#55](https://github.com/labscommunity/cascadia/issues/55)) ([b4a58bc](https://github.com/labscommunity/cascadia/commit/b4a58bc309f709661aac66bc876a0f2985b42e8b))
* **ov-optimum:** graceful fallback when transformers spec decode incompatible ([ad2ee5c](https://github.com/labscommunity/cascadia/commit/ad2ee5cabce205d42548175f3dcfbc160a64713c))
* **ov-runtime:** use HF cache only; tolerate stale bundled tokenizer ([675124b](https://github.com/labscommunity/cascadia/commit/675124b4f1c9096be07b7aca00c24c31108b34ce))
* **placement:** ignore non-finite latencies; lock tie-break/single-device/NaN with tests ([fd19133](https://github.com/labscommunity/cascadia/commit/fd19133036d91a7727f127136512521a8d4c58af))
* **placement:** kill spawned workers if the launcher errors (no orphans); scope --relay-host to local ([18b19d9](https://github.com/labscommunity/cascadia/commit/18b19d9450fb7ccebc7192f494cab6076086274a))
* **placement:** prefer slice::contains in the per-stage profiler (clippy) ([5898f4e](https://github.com/labscommunity/cascadia/commit/5898f4ec047950fd24d9caca5cbe6b5cb86749a5))
* **placement:** profiler falls back on an unreadable device mem budget; validate --mem-headroom/--pool-gb ([0026465](https://github.com/labscommunity/cascadia/commit/002646559b0ec235a67185184b567b2a6e51f528))
* **placement:** warn on memory-exhausting placements ([#67](https://github.com/labscommunity/cascadia/issues/67) was not corruption) ([e88eff5](https://github.com/labscommunity/cascadia/commit/e88eff56fea6ec010c972bfa5a12497c3921e62a))
* **placement:** warn when a placement's footprint will exhaust RAM and swap ([#67](https://github.com/labscommunity/cascadia/issues/67)) ([e440333](https://github.com/labscommunity/cascadia/commit/e440333265bb0eae3e87cb131fc20d2543246b01))
* **qwen36:** f32 inference precision for the MoE router ([96e7888](https://github.com/labscommunity/cascadia/commit/96e7888fd6f9cf3fb8acfc171d423d0b73802cd8))
* **qwen36:** fail loud on handshake NAK instead of silent empty 200 ([2d3b7da](https://github.com/labscommunity/cascadia/commit/2d3b7da587d45e14051fa45dcbcd4b45d4e61223))
* **qwen36:** fail loud on mid-generation backend errors ([375b0f3](https://github.com/labscommunity/cascadia/commit/375b0f3c996cc53dd1ee0340a231134c454a4a49))
* **qwen36:** harden pipeline driver — empty-prompt, reply timeout, queue cap ([6c4ba86](https://github.com/labscommunity/cascadia/commit/6c4ba8621e3f77b8d992b37f29f9e8f02d44e100))
* **release:** bundle the OpenVINO redistribution notices from docs/licensing ([7eda0b6](https://github.com/labscommunity/cascadia/commit/7eda0b6ab2f0e7d8a3892762edb68431c419b3a5))
* **release:** keep dev/debug SDK files out of the runtime bundles ([e48959f](https://github.com/labscommunity/cascadia/commit/e48959f437d1cd31dd8913c0000b761a25fb1c14))
* **release:** make the Linux bundle rpath cover transitive OpenVINO deps ([60d313d](https://github.com/labscommunity/cascadia/commit/60d313d32599d7f1a6200f511346ef49d2b38fd1))
* **runner,cli:** relay loop exits on dead peer link instead of spinning ([fe049f6](https://github.com/labscommunity/cascadia/commit/fe049f62ac1589601fc345717f2ddafd0da4d089))
* **runner,engine:** throttle relay loop on persistently-failing step() ([9582067](https://github.com/labscommunity/cascadia/commit/9582067a6a1a4bfa92aa05ffa2bd0c84bfe6bb0b))
* **runner:** bound ChunkStream cross-task spin on repeated SAME task, not any foreign error ([355fb8e](https://github.com/labscommunity/cascadia/commit/355fb8e3d046083dd216551cb37369cb35826ae3))
* **runner:** bound cross-task error continue in ChunkStream ([925c941](https://github.com/labscommunity/cascadia/commit/925c941737c5a824a4cc3ed1fa8b8cd8bdf3c2cf))
* **runner:** surface engine step failures to clients as final error chunks ([73d008e](https://github.com/labscommunity/cascadia/commit/73d008eb13f2da7325e42bcb237b23cab7b69e95))
* **runtime:** stop on any of the model's eos_token_ids (not just the first) ([1bb10d7](https://github.com/labscommunity/cascadia/commit/1bb10d74d4b68778651c0bcca4f2ab53c02e60f3))
* **rust:** build.rs uses lib/intel64/Release on Windows; add Linux fallback ([81b747d](https://github.com/labscommunity/cascadia/commit/81b747df6b596cd9f0bb22c41739d4d6f7c1712f))
* **rust:** dist-spec hidden_states must be f16; expose port aliases; +7 tests ([16d7c97](https://github.com/labscommunity/cascadia/commit/16d7c975d38ce2326cd39db8c166eef2a1480868))
* **shard:** assert static NPU shapes post-compression + guard fp32 (review [#2](https://github.com/labscommunity/cascadia/issues/2)) ([2482502](https://github.com/labscommunity/cascadia/commit/2482502f74e1a182bbfd5a7ccc0f28b9292e9689))
* **shard:** check gemma-4 IR guards before importing the surgery module ([fde7171](https://github.com/labscommunity/cascadia/commit/fde7171c6a3cc9e41099699b8036f43a0cdc4802))
* **shard:** fail loudly when an OV VLM IR dir has an unreadable config.json ([75754f6](https://github.com/labscommunity/cascadia/commit/75754f60e475aaaf57ea35fbec686fa94ea7a9f2))
* **shard:** load exported shards on the OpenVINO CPU plugin via init-bearing stateful KV ([2f8ac64](https://github.com/labscommunity/cascadia/commit/2f8ac64b60d830214e29eb36d08e54d91dab4b2b)), closes [#57](https://github.com/labscommunity/cascadia/issues/57)
* **shard:** load exported shards on the OpenVINO CPU plugin via init-bearing stateful KV ([#57](https://github.com/labscommunity/cascadia/issues/57)) ([ac9616f](https://github.com/labscommunity/cascadia/commit/ac9616fed536fb62a26e85a71568133d6f62ea8a))
* **shard:** pin static NPU output/hidden shapes + guard static-seq ([#37](https://github.com/labscommunity/cascadia/issues/37)) ([706f3cd](https://github.com/labscommunity/cascadia/commit/706f3cd6e53176c4d1a43343899040433ffabe33))
* **shard:** read rope_theta from rope_parameters dict (transformers 5.x) ([11b77dc](https://github.com/labscommunity/cascadia/commit/11b77dcad5df24f489cac1e31bcd17c8f838190b))
* **shard:** reject --target npu / --layer-split on gemma-4 IR path + forward --stage ([de830c6](https://github.com/labscommunity/cascadia/commit/de830c60bcfb2555a5613171859ecc3a5c6a1617))
* **shard:** review hardening — gemma class selection, MoE/.bin/rotary robustness ([2906820](https://github.com/labscommunity/cascadia/commit/2906820ce8f3a0554914296bf7dd070263152bfb))
* **shard:** say that --quantization is inherited on the gemma-4 IR dispatch ([91af142](https://github.com/labscommunity/cascadia/commit/91af1427f8072d43565f5f0eb92922000db297e9))
* **shim:** pass explicit monostate streamer to VLMPipeline::generate ([0499ad1](https://github.com/labscommunity/cascadia/commit/0499ad1a9cd51912b98d0b835e96953d60dd5121))
* **shim:** pass integer NPU LLM properties as int64, not string ([a44c74f](https://github.com/labscommunity/cascadia/commit/a44c74f4c99bd72bcfa29a5aed9368c22ad35dff))
* **shim:** reject partial-integer NPU values; test the int64 coercion ([3b535b5](https://github.com/labscommunity/cascadia/commit/3b535b55b75445c72a2938881d43ffd8231b3d5a))
* **sparse-moe:** clamp corrupt zero repetition_penalty on the wire ([#14](https://github.com/labscommunity/cascadia/issues/14) review) ([5a5155b](https://github.com/labscommunity/cascadia/commit/5a5155b84b2829ab0890771c55d75885b7a973f5))
* **sparse-moe:** only set SNIPPETS_MODE on CPU (unblocks iGPU pipeline rank) ([34c5fff](https://github.com/labscommunity/cascadia/commit/34c5fff1da5213e036ac75ac7915b6a453d1a095))
* **sparse-moe:** SNIPPETS_MODE=DISABLE — the real MiniMax-M2 coherence fix ([2ecbd54](https://github.com/labscommunity/cascadia/commit/2ecbd547478218b5e88e400526a72d7be2a5bc89))
* **sparse-moe:** surface connection-fatal Err on latched worker disconnect ([a64b597](https://github.com/labscommunity/cascadia/commit/a64b597da7bf88ddcfd8d75618a75d4282b0a3fe))
* **test:** avoid clippy::approx_constant deny — replace 3.14 with 3.5 in dtype roundtrip test ([a1beed7](https://github.com/labscommunity/cascadia/commit/a1beed737fbd6129b11847417a0448b83650b538))
* **tool-calling:** enable minijinja json feature for tojson filter ([0abdb4c](https://github.com/labscommunity/cascadia/commit/0abdb4cc6b0411044c264c65879a4fc92a1633c9))
* **transport,engine:** classify peer crash (RST) + mid-frame timeout as connection-fatal ([5dd7ad0](https://github.com/labscommunity/cascadia/commit/5dd7ad07d3fdf0cd18ebbbcf5490625b9c5c6018))
* **transport:** deadline mid-task replies — idle-tolerance must not cover in-flight work ([24ccd95](https://github.com/labscommunity/cascadia/commit/24ccd951a6dabdbf92a83170949786fbd57662be))
* **transport:** deadline mid-task replies to prevent chain-head wedge ([9c12347](https://github.com/labscommunity/cascadia/commit/9c123478257da6ad47fe31688cc6973a5b592041))
* **transport:** floor the idle ceiling at the activation recv timeout ([f2ef4c8](https://github.com/labscommunity/cascadia/commit/f2ef4c827f917b55c054875db3f0c61ca7578c0b))
* **transport:** idle wait for the next frame must not time out ([cdb628d](https://github.com/labscommunity/cascadia/commit/cdb628da3a607bb7c596407a0993361d9cdfe450))
* **transport:** idle-ceiling fire is connection-fatal ([10d3adb](https://github.com/labscommunity/cascadia/commit/10d3adbe871bc273975b3ad71a34bf95b6f9df07))
* **transport:** mid-frame recv stall is connection-fatal ([71a098a](https://github.com/labscommunity/cascadia/commit/71a098af26a021ec2b1b7f3faff434e84368352f))
* **transport:** poison connection on failed reply; widen prefill reply budget ([632683a](https://github.com/labscommunity/cascadia/commit/632683a2caa9e96c9ea16fc4ad39fc58ebf91145))


### Performance

* AXPY-form sparse FFN down + on-disk transposed-weight cache ([#43](https://github.com/labscommunity/cascadia/issues/43)) ([08fbd35](https://github.com/labscommunity/cascadia/commit/08fbd3593805897214465a900ea8c20a58488f91))
* **dist-spec:** async overlap target wire with draft compute (+24% throughput) ([d7e5139](https://github.com/labscommunity/cascadia/commit/d7e513945f2797c297c5551f54affecaf82d4854))
* **dist-spec:** async overlap target wire with draft compute (+9% long-gen, +19% K=1) ([b12105b](https://github.com/labscommunity/cascadia/commit/b12105b1fd08960f2c57d906c73c5ee83756c998))
* per-channel CHESS FFN sparsity thresholds ([#38](https://github.com/labscommunity/cascadia/issues/38)) ([#44](https://github.com/labscommunity/cascadia/issues/44)) ([3aa7548](https://github.com/labscommunity/cascadia/commit/3aa7548f8e6e855082f21c62acc6b9a99c87d629))
* PowerInfer port — bounded LRU + two-phase Gate-first FFN sparsity ([#34](https://github.com/labscommunity/cascadia/issues/34)) ([cde70be](https://github.com/labscommunity/cascadia/commit/cde70be11b553c0013f8d282492d4f9270b8c41b))
* **rust:** add ov-dist-spec warmup to pre-pay cold init; Rust now beats Python ([260566e](https://github.com/labscommunity/cascadia/commit/260566efb6ad07e10d693fa6dd64513c2124c130))
* **rust:** drop f16 round-trip in worker step + document distributed perf gap ([b396709](https://github.com/labscommunity/cascadia/commit/b396709d3e9d8dbb5e7b2dae8945bdde22324b41))
* **rust:** release-mode build is 1.75x faster on dist-spec; close 2.2x gap to 1.29x ([655f6b8](https://github.com/labscommunity/cascadia/commit/655f6b8c5da261a56d681c4d47ebe4fc2d479ff7))
* **sparse-moe:** A8 KV bf16 + C1 prefetch + SIMD multi-token int4 + spec-decode (+60% vs main) ([72561f0](https://github.com/labscommunity/cascadia/commit/72561f0e0a778db00724c183b32af23fde9abff6))
* **sparse-moe:** add --top-k-override + --routing-threshold flags (A3) ([#29](https://github.com/labscommunity/cascadia/issues/29)) ([200ec2f](https://github.com/labscommunity/cascadia/commit/200ec2f41d90000b12b6d698df2c6878c8c6b8e6))


### Refactor

* **cli:** compare ShardDtype with != instead of !matches! ([5e04833](https://github.com/labscommunity/cascadia/commit/5e04833e39f4e143052ad686ef8f0ccbc83efa35))
* **cli:** dedupe static-context default into a const + ShardDtype Eq ([7eac4d1](https://github.com/labscommunity/cascadia/commit/7eac4d1d0b6f7b53045def990f6a3745457cae54))
* **cli:** engine registry + supervisor integration ([8cbfdc4](https://github.com/labscommunity/cascadia/commit/8cbfdc46600459cdb2ebe6c9046df73bf5da3a00))
* **engines:** shared HF hub helpers ([262fe4d](https://github.com/labscommunity/cascadia/commit/262fe4d5e6f18171963bb2baf5a2d3af3c8ddef1))
* **gemma4:** drop dead fields inherited from the v5 clone ([fea2bb1](https://github.com/labscommunity/cascadia/commit/fea2bb1dbd5718111cf4e7f6b0f9bfcdd0fc7985))
* **gemma4:** source-id cross-KV pairing, position-on-wire, drop static path ([429d88f](https://github.com/labscommunity/cascadia/commit/429d88f32da5f9fa8fc5b7f1231055fa33617a2c))
* **runtime:** harden + optimize the static-KV NPU path (review [#2](https://github.com/labscommunity/cascadia/issues/2)) ([b01f4fa](https://github.com/labscommunity/cascadia/commit/b01f4fa7874f64d64523acb16834a61b9eb81c17))
* **sparse-moe:** layer-range CLI flag + split-path test (replace env hack) ([6503fa0](https://github.com/labscommunity/cascadia/commit/6503fa09dc6cd6dd37cca0e6ba04c39d1f78d95c))


### Documentation

* add SECURITY.md with threat model and reporting policy ([ff40b43](https://github.com/labscommunity/cascadia/commit/ff40b4305636fa1ec0b814398f09ae3c7448d9d9))
* add Tahoma logo to README header ([ab64a2a](https://github.com/labscommunity/cascadia/commit/ab64a2aba07e045d39150c35df30ab6640455c8b))
* **arch:** Mistral family deep-dive (mistral, NeMo, Mistral-Small 3.x) ([c92ba71](https://github.com/labscommunity/cascadia/commit/c92ba717426cbfdf0b566398952152b1465043b2))
* **arch:** Mistral family deep-dive (mistral, NeMo, Mistral-Small 3.x) ([2a1a6bc](https://github.com/labscommunity/cascadia/commit/2a1a6bc11747c7f4594ac595ed38f23b45c0a002))
* **arch:** per-family support audit + architectures/ deep-dives ([fbd81d5](https://github.com/labscommunity/cascadia/commit/fbd81d506fbd7e5028bc5290549da21cf22a64d8))
* **arch:** per-family support audit + architectures/ deep-dives ([3c8c610](https://github.com/labscommunity/cascadia/commit/3c8c6105d06f18ca7fd51acd138c8e3f727e41e7))
* bump OpenVINO GenAI SDK examples to 2026.2.0.0 ([2202524](https://github.com/labscommunity/cascadia/commit/22025249f1189f130f55ff451c5213a4406ee07b))
* **ci:** fix pin-comment wording, expand go-public security checklist ([ee13ecd](https://github.com/labscommunity/cascadia/commit/ee13ecdf4e42a617bf5ef4cf39d1af491508f258))
* clean up README.md and formatting improvements ([99468e2](https://github.com/labscommunity/cascadia/commit/99468e2efdb0ec64e46a76744729a60a7f786f2c))
* clean up README.md and formatting improvements ([88bfb25](https://github.com/labscommunity/cascadia/commit/88bfb256eec5595866718330b1f728db4b5f8855))
* **cli:** document full OpenVINO --device string forms ([#12](https://github.com/labscommunity/cascadia/issues/12)) ([c0c7bd7](https://github.com/labscommunity/cascadia/commit/c0c7bd702b703e3c3fb922977d8c930e7cc5128f))
* **cli:** document full OpenVINO --device string forms ([#12](https://github.com/labscommunity/cascadia/issues/12)) ([5ac5b88](https://github.com/labscommunity/cascadia/commit/5ac5b884337aed6daa51e8add2efd9e60105e622))
* correct NPU flag help + sparse-moe property-scope comment ([8a8118e](https://github.com/labscommunity/cascadia/commit/8a8118eb47fc4f002f857b5b712db21c930c7bf6))
* correct overclaims from the fix pass; rationalize docs layout ([eb03f88](https://github.com/labscommunity/cascadia/commit/eb03f88af27391908aa12afbc5445b1ca5018a7e))
* **dist-spec:** correct relay-loop-exit comment ([4b11927](https://github.com/labscommunity/cascadia/commit/4b119270352b64379dc496d9d0435f4d1e6f0d98))
* document MiniMax-M2 support, constraints, export/run ([1ffc154](https://github.com/labscommunity/cascadia/commit/1ffc154fcd99cfe65ecef75016812cf174f0551a))
* document prebuilt release bundles and the release process ([92fbca1](https://github.com/labscommunity/cascadia/commit/92fbca1a6fd10cbcb58fb92863beae1cb1f50b30))
* drop H1, remove stale Phase 12 line, fix cluster section ([544cbcd](https://github.com/labscommunity/cascadia/commit/544cbcda97a547ca1188ff54e64fa791f77b1151))
* drop the remaining em-dashes from README ([2f37d5a](https://github.com/labscommunity/cascadia/commit/2f37d5a3a7bed8e274a9c1475cc2950ac34dcce0))
* **engine,runner:** cancel contract as implementor obligation; relay-loop doc matches behavior ([97c9a93](https://github.com/labscommunity/cascadia/commit/97c9a93ce0c6eaed06ef7b7e77181f3f9561262d))
* **engine,runner:** step Err attribution, failure idiom, relay-loop throttle obligation ([b9347a9](https://github.com/labscommunity/cascadia/commit/b9347a9cdc178188ebb5d5fb9128fc5dc43f7551))
* exporter polish record (embeds-free mid stages, logit slice, 8/8 multi-token) ([62c6ab0](https://github.com/labscommunity/cascadia/commit/62c6ab000a47f56e3acbf54c684944530cc0f922))
* fix parallel-review findings — dangling private-commit/spec refs, engine list ([f58631f](https://github.com/labscommunity/cascadia/commit/f58631f076f5992ffb69dcb82aaafc073bb0380d))
* fix post-scrub review findings ([2f95d78](https://github.com/labscommunity/cascadia/commit/2f95d789aa081c5940d2e6620794684ded13d094))
* fix scoped-review findings ([9ad15c5](https://github.com/labscommunity/cascadia/commit/9ad15c54d2cf487a30002867848ffa6c3796e620))
* fix two typos in the README cleanup ([ef5f7dd](https://github.com/labscommunity/cascadia/commit/ef5f7dd4f471a976d7d0c2721bc1c8fa746c8a3f))
* **gemma4:** fix comment inaccuracies flagged in review ([4eb7d65](https://github.com/labscommunity/cascadia/commit/4eb7d652fb4553aec25c6333820b147828f6d354))
* **gemma4:** record validated max_ops=64 rationale + base-vs-it source lesson ([335f6d1](https://github.com/labscommunity/cascadia/commit/335f6d1deb4377263d4b7a526aad652050962e58))
* **gemma4:** scrub internal node/prototype-script references from surgery tool ([def60d5](https://github.com/labscommunity/cascadia/commit/def60d5bd7a96ba10f2837682aa871fb6f2d6377))
* M3' prototype results — 64/64 parity, heterogeneous device map ([2998d7b](https://github.com/labscommunity/cascadia/commit/2998d7b8250eff4177e9a62a9638173cc15e8eb3))
* M3' robustness validated (streaming cadence, cancel, disconnect; E2E 5/5) ([aa2a465](https://github.com/labscommunity/cascadia/commit/aa2a465e7c20a11d4ce066587d78a2e2db4d3000))
* M3' TTFT record — chunked batched prefill shipped, GPU split ruled out ([b395c6f](https://github.com/labscommunity/cascadia/commit/b395c6f04e3137c798cb7fcddaeac1d07d4afec2))
* **minimax-m2:** correct root cause — SNIPPETS bug, not expert quant ([fadc5fc](https://github.com/labscommunity/cascadia/commit/fadc5fca51ba5ea3488f9b15211ba0d2c9733434))
* **minimax-m2:** expert-backend perf comparison + output-quality findings ([ff7c4eb](https://github.com/labscommunity/cascadia/commit/ff7c4eb26f225e0eba564558028a18af59dffa8c))
* **minimax-m2:** final perf — int4_bin+SNIPPETS-fix is fast AND coherent ([b4d690f](https://github.com/labscommunity/cascadia/commit/b4d690f62ef327128c6e1e1614193d5181bb9da2))
* move CI badge under the Status heading ([824299d](https://github.com/labscommunity/cascadia/commit/824299d58a23ccaf660127e66f35c3be56cb3661))
* onboarding overhaul — INSTALL/QUICKSTART/CONTRIBUTING, setup scripts, Docker ([b01234e](https://github.com/labscommunity/cascadia/commit/b01234efe006b021aefb12813fd26587ab087199))
* ov engines docstring summary (4 engines now) ([9b68415](https://github.com/labscommunity/cascadia/commit/9b68415ab57b4be83763caea40f4a1a6b811bd69))
* **ov-genai:** record on-HW seed-reproducibility limitation ([#14](https://github.com/labscommunity/cascadia/issues/14)) ([e525485](https://github.com/labscommunity/cascadia/commit/e52548530f5f586cd0b39f76112744211859a8fc))
* **perf:** correct PERFORMANCE.md claims flagged in review ([6d9c3f4](https://github.com/labscommunity/cascadia/commit/6d9c3f4e1f4d848fd0858a17f0e174a036f4a67b))
* **placement:** [#41](https://github.com/labscommunity/cascadia/issues/41) design — measured hardware, regime analysis, ILP plan ([2cac67c](https://github.com/labscommunity/cascadia/commit/2cac67c41b02588095e939e45719b5ad1fafa6e1))
* **placement:** correct overflow-tier wording + shim input-shape comment ([026a70b](https://github.com/labscommunity/cascadia/commit/026a70bebf254ef60077fa78af52dc4e1ab017aa))
* **placement:** record three-regime benchmark + the UMA-spill finding ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([40bce33](https://github.com/labscommunity/cascadia/commit/40bce33dffcd9d58524fe96c83fe9fc8634e86e1))
* **placement:** update for implemented profile-stages/place/run-placement ([#41](https://github.com/labscommunity/cascadia/issues/41)) ([6245212](https://github.com/labscommunity/cascadia/commit/6245212b8704e39670ed3817b2a448121baee847))
* qwen3_5_moe alias comment + SHARDING.md row reflect staged-shard support ([b895ac1](https://github.com/labscommunity/cascadia/commit/b895ac12eb6607e54b6d17778163faf001044ec3))
* Qwen3.6-35B-A3B sharded-support design spec (draft) ([6c7eff8](https://github.com/labscommunity/cascadia/commit/6c7eff8296d082ba110b68c1a9ab33e2b16be0ff))
* qwen3.6.md — CPU validation, device smoke matrix, OVMS 2026.2 install ([1166782](https://github.com/labscommunity/cascadia/commit/11667822aa39126c1eb81f8c8ab1175921ce4679))
* qwen3.6.md — hardware validation record (pawan-01, 2026-06-11) ([92e5020](https://github.com/labscommunity/cascadia/commit/92e502014deb1dbae5176a1f4dbcb79daa28aceb))
* qwen36 spec — M2'-0 PASSED, strategy C (all-CPU OV-IR experts) ([3a82cdf](https://github.com/labscommunity/cascadia/commit/3a82cdfa39182248eb628c57f265050907cb3a67))
* qwen36 spec rev 2 — apply adversarial + feasibility review ([78011f6](https://github.com/labscommunity/cascadia/commit/78011f691782f756583a289fc3bbf465f58b8a16))
* qwen36 spec rev 5 — sparse-moe port now conditional on ISA/device probe ([b593466](https://github.com/labscommunity/cascadia/commit/b593466b0be85ff3afdf66b1147a8019752a569c))
* qwen36 spec rev 6 — full rewrite from 4-angle review (incl. Codex) ([fdcb1aa](https://github.com/labscommunity/cascadia/commit/fdcb1aaac7e17a89a4dc1a545bb7cb412208a16d))
* **qwen36:** drop dev-spike orphans, trim spec process-meta, add surgery README ([b21052e](https://github.com/labscommunity/cascadia/commit/b21052ee89ff124981ccf1bb895c3361ff97834e))
* **r1-distill:** pipeline-parallel verified on iGPU+iGPU AI PCs ([50717e8](https://github.com/labscommunity/cascadia/commit/50717e89080c8c95a7aec58b9a4eb8b175a22ecc))
* README logo + accuracy pass ([276d97b](https://github.com/labscommunity/cascadia/commit/276d97b6df2ffb09f1abb0bad1bf235ff8daf157))
* README quick start + per-engine + deploy guides ([3c609d2](https://github.com/labscommunity/cascadia/commit/3c609d26dcf959e7afae01450f52abcfce110a60))
* **readme:** sync engines, crates, and tooling with current main ([50dcfef](https://github.com/labscommunity/cascadia/commit/50dcfef928ed63b1446738a097672ce501ad58b5))
* remove ONBOARDING_RESEARCH.md (moved to rainier) ([c66dc01](https://github.com/labscommunity/cascadia/commit/c66dc0180dd5a272b7600b4214fe0084d00d5bc7)), closes [#52](https://github.com/labscommunity/cascadia/issues/52)
* replace fleet hostnames with neutral node labels ([a3c81b5](https://github.com/labscommunity/cascadia/commit/a3c81b5236b82706f011b39e41c3f1c81b0716e0))
* restructure README into standard OSS layout ([1912bfb](https://github.com/labscommunity/cascadia/commit/1912bfbc178e747bbfd6b43bc655626dbf808f14))
* **rust:** record Phase 14 perf validation (28.75 tok/s, no regression) ([c07f20c](https://github.com/labscommunity/cascadia/commit/c07f20c75c68e1ce7f8cd7849e063892456cd6b4))
* **rust:** STATUS.md updated with release-mode perf data + remaining-gap analysis ([72eddfa](https://github.com/labscommunity/cascadia/commit/72eddfa93d7480cec1beff5ae883e839cacfc21f))
* scrub internal references for public release ([7fc4f1a](https://github.com/labscommunity/cascadia/commit/7fc4f1a5bd800bdea4ddf3b250046713ebc49f77))
* set code of conduct enforcement contact ([9599719](https://github.com/labscommunity/cascadia/commit/95997190d35d7184cf7a2c5a071b1def02244390))
* **sharding:** add NPU sharding guide ([cc2d740](https://github.com/labscommunity/cascadia/commit/cc2d7405259f1786609990f6beeb14cdedd2fbee))
* **sharding:** add NPU sharding guide (static/stateless export + host KV ring) ([6baae49](https://github.com/labscommunity/cascadia/commit/6baae49c8e6d025d18478df3da61fbb794fe4128))
* **status:** refresh STATUS.md to current state ([abcda69](https://github.com/labscommunity/cascadia/commit/abcda693b3d7f983f99e97add0ad0c2c0e7b577a))
* tool-calling validated on both serving paths ([#77](https://github.com/labscommunity/cascadia/issues/77) Path-1 item) ([8f10aee](https://github.com/labscommunity/cascadia/commit/8f10aee735839d366a3356222c8908490a54b0b5))
* **transport, engine:** correct idle-wait and prefill-budget claims ([1da7cd3](https://github.com/labscommunity/cascadia/commit/1da7cd3c4fdc12d75115ea401272a8b7bed75d7b))
* **transport:** clarify frame-start wait is ceiling-bounded, not unbounded ([44fe5fc](https://github.com/labscommunity/cascadia/commit/44fe5fc4e659b0e1edc650b2f82c241d169bd8ce))
* **transport:** note recv_exact timeout is wall-clock, not idle ([f58b1b0](https://github.com/labscommunity/cascadia/commit/f58b1b0affb7fb7515fc2cae75b34e42356f95fe))


### Testing

* add 44 tests covering runner, api, registry, builders, cli, protocol ([2c7ab02](https://github.com/labscommunity/cascadia/commit/2c7ab020a4b32d60a9a6640208a35db6d93d52b3))
* **api:** brace-balance parser stress (M2) ([d6f61e4](https://github.com/labscommunity/cascadia/commit/d6f61e47ce45b6f7d2dcd50b960e92907998d4b9))
* **api:** cover completions length + engine-error; note legacy logprobs shape ([#19](https://github.com/labscommunity/cascadia/issues/19) review) ([2432d14](https://github.com/labscommunity/cascadia/commit/2432d14c6d633f87720317fd3e7224f9004299fc))
* bless qwen36 parity golden (64 tokens, batched-prefill engine on pawan-01) ([b68b4a6](https://github.com/labscommunity/cascadia/commit/b68b4a65257cf9728d2955139256958fdb52eee3))
* **cli:** pin the full NPU exporter argv as a golden vector ([4d282c0](https://github.com/labscommunity/cascadia/commit/4d282c0ab4c81f3e82c7cefdbdf5bb382da1a7f3))
* **e2e:** address multi-node pipeline review nits ([fe8ab43](https://github.com/labscommunity/cascadia/commit/fe8ab43832e148f6d550ccd42482d0fea7b155c9))
* **e2e:** multi-node pipeline review follow-ups ([145694c](https://github.com/labscommunity/cascadia/commit/145694c67ba57d86caee516b77390773fc51d280))
* **e2e:** parameterize cross-node test by engine/model/device ([7516544](https://github.com/labscommunity/cascadia/commit/751654401e83a91ce4051f7c2268efee3f6a8904))
* **e2e:** sharded multi-stage pipeline tests (loopback + cross-node via fleet) ([0a2f2cd](https://github.com/labscommunity/cascadia/commit/0a2f2cd799409bcdbf31e848ea6ca173e6c41c6a))
* **e2e:** sharded multi-stage pipeline tests (loopback + cross-node via fleet) ([e27dfb6](https://github.com/labscommunity/cascadia/commit/e27dfb6c253f69c031b3f488bc30029645451872))
* **export:** --tiny-layers to size the synthetic M2 for pipeline tests ([a98cd7f](https://github.com/labscommunity/cascadia/commit/a98cd7f15f69246c009ff754a01a40c737b12d4b))
* **gemma4:** hermetic load-guard test + clippy is_multiple_of ([8ab4b11](https://github.com/labscommunity/cascadia/commit/8ab4b11bf69e4e3ef855ca87e9b2c3a52205afbf))
* **gemma4:** pin dispatch fail-fasts and the surgery tool's pure seams ([3663987](https://github.com/labscommunity/cascadia/commit/36639876148cf3eef9e9f329c0b1af492939d274))
* **minimax-m2:** add --ov-experts probe mode (matches docs) ([33cf599](https://github.com/labscommunity/cascadia/commit/33cf599dcc719ae39d6ff0874eccc2efdb1a4033))
* **openvino:** assert OV perf props reach the PluginConfig ([bbe0985](https://github.com/labscommunity/cascadia/commit/bbe0985c28e4785f7bdfc4b121f7493f61275b3f))
* qwen36 greedy-parity regression gate (ignored; needs shards on node) ([dc003cb](https://github.com/labscommunity/cascadia/commit/dc003cb1dd5cfba7f8fc5c0d2254ea11e8fc65ab))
* **rust:** add tests-e2e integration crate + STATUS doc updates ([405f3f9](https://github.com/labscommunity/cascadia/commit/405f3f92f2971a232520800a82ad493cdec1d66e))
* **sparse-moe:** real-model MiniMax-M2 generation smoke test ([e03a601](https://github.com/labscommunity/cascadia/commit/e03a601250190482873cc1cf1f79340ac9da5bd8))
* **tp_group:** hold ephemeral discovery sockets simultaneously ([b191069](https://github.com/labscommunity/cascadia/commit/b1910699dcba21ff8464c849bebba9c9d275a993))
* **transport:** pin ActivationClient reply deadline and poison behavior ([c3720c8](https://github.com/labscommunity/cascadia/commit/c3720c8190ef0820ba2c82ec1a5eb9e02feb0806))
* **transport:** stress tests for idle-tolerant frame-start recv ([459832f](https://github.com/labscommunity/cascadia/commit/459832f6f5b301bc0098504d25c68b797b0f4e05))


### CI

* add AI-PC tests workflow (OpenVINO tier on self-hosted runners) ([818351a](https://github.com/labscommunity/cascadia/commit/818351a96d6e986f6b427e92eb7cf0093b9ce861))
* add AI-PC tests workflow (OpenVINO tier on self-hosted runners) ([9c0c731](https://github.com/labscommunity/cascadia/commit/9c0c731c82866141cb3ace08e953496ffd0183ee))
* add cargo-deny supply-chain gate ([b614871](https://github.com/labscommunity/cascadia/commit/b614871cd47174af4c2d6525eef9261d0417beae))
* **ai-pc:** run cascadia-ov-genai-shim openvino tests on the runner ([1a43969](https://github.com/labscommunity/cascadia/commit/1a439690d0238d987fb4a608a234a7d9c0ef8d43))
* build tahoma binary before test step (e2e tests exec it) ([320035c](https://github.com/labscommunity/cascadia/commit/320035c163a1d43f9db6148b118b26f8c02f5dd5))
* Cargo.lock — minijinja `json` feature pulls serde_json (cf94415 follow-up) ([b823acc](https://github.com/labscommunity/cascadia/commit/b823acc13b55e7f6ae9a332ddca25c589b73aeb4))
* github actions for ruff + pytest on py3.11/3.12 ([35ded04](https://github.com/labscommunity/cascadia/commit/35ded04c9e9b101fc83b193af54d66671d4e9e5d))
* harden workflows with least-privilege perms, SHA-pinned actions, timeouts ([607b5c5](https://github.com/labscommunity/cascadia/commit/607b5c5add7819f3d9e201c729651ff980d5aecb))
* **release:** automate versioning and releases with release-please ([0da6c82](https://github.com/labscommunity/cascadia/commit/0da6c822b68007dffdb23df7391bcc0f9e53bf54))
* **release:** build cascadia OpenVINO binary bundles for Linux + Windows ([cba4b38](https://github.com/labscommunity/cascadia/commit/cba4b380e498f82f41f569416b199cec39d2999f))
* **release:** cascadia binary bundles + release-please version automation ([c515fe8](https://github.com/labscommunity/cascadia/commit/c515fe8458325a1708fde994e4a5fddcecd007c2))
* scope security-workflow concurrency by event ([921a780](https://github.com/labscommunity/cascadia/commit/921a7807e89a6a2fff0587e1a59fbe17910ecdb0))
* security hardening (perms, SHA-pinned actions, cargo-deny gate) ([f46a495](https://github.com/labscommunity/cascadia/commit/f46a495a4503702dc7c2762a5a5d7df4d05abd5b))


### Miscellaneous

* [#77](https://github.com/labscommunity/cascadia/issues/77) Part A remainder — 2026.2 floor, dead deps, exporter pins ([64992a4](https://github.com/labscommunity/cascadia/commit/64992a40314c9e47850bb8d71c4d115206c2e8f8))
* add community health files ([280b5ce](https://github.com/labscommunity/cascadia/commit/280b5ce090f8ea4f5ea51702ddf2093e65805e6f))
* add OpenVINO GenAI 2026.2 support ([30c8d29](https://github.com/labscommunity/cascadia/commit/30c8d29419a16fa53a6f1523b148ecf2ab8a9216))
* cargo fmt ([a0d0a27](https://github.com/labscommunity/cascadia/commit/a0d0a2784d7bb02794682e916970aeacd8db5025))
* **deny:** drop unused license allowances, fix policy comments ([0151268](https://github.com/labscommunity/cascadia/commit/015126800c3df68169237bf9fd06bffa476e0ce4))
* **deps:** bump anyhow/memmap2 to patched releases, drop stale deny ignores ([7954ac5](https://github.com/labscommunity/cascadia/commit/7954ac52e8b872cd29249e649830ef86887930ac))
* **deps:** pin transformers &lt;5.0 ([c908233](https://github.com/labscommunity/cascadia/commit/c908233c7006b8bde02fc3808d672517322775a8))
* **deps:** update crossbeam-epoch to 0.9.20 (RUSTSEC-2026-0204) ([f61b4de](https://github.com/labscommunity/cascadia/commit/f61b4de9b4583ebdc662fa133ed3f980da111695))
* gitignore .claude/ (Claude Code local state + worktrees) ([671bb27](https://github.com/labscommunity/cascadia/commit/671bb27522f60d24ba8c773baa8d3ecab7732afa))
* hoist rust/* to repo root (no more rust/ subdirectory) ([c70fc18](https://github.com/labscommunity/cascadia/commit/c70fc18f64420e70e1ef1aef23af77f1006b9046))
* mark workspace crates publish = false, promote wildcard ban to deny ([21b5efa](https://github.com/labscommunity/cascadia/commit/21b5efae22eee21ba213135fd9575ccc4717a178))
* **openvino:** demote step_first's inner failure WARN to debug ([8a646cb](https://github.com/labscommunity/cascadia/commit/8a646cbaffd25e242492257c4ea67c482b95683c))
* **openvino:** hoist warn_limit import + correct module-doc scope ([d9378f2](https://github.com/labscommunity/cascadia/commit/d9378f2952a9c0ca0d3b79b013fa310f607af701))
* **ov-spec:** default K=4 (sweet spot on Arc B390) ([f2789e5](https://github.com/labscommunity/cascadia/commit/f2789e50bd9fe33f6a3ec046fb66fad0b37425aa))
* public-release repo cleanup — license, docs scrub, community files ([1927898](https://github.com/labscommunity/cascadia/commit/19278983bcd61ce644551a7a627980b1c98d7376))
* remove agent instructions file; ignore local AI tool state ([f9b62cc](https://github.com/labscommunity/cascadia/commit/f9b62cc40aa8db763cafc425fa3a3d230c5ad6c1))
* remove Python tree (Phase 12) — Rust port is the sole impl ([60bfe91](https://github.com/labscommunity/cascadia/commit/60bfe91c02947c2ebfcb7f34ac3c05aabe05bfe6))
* rename tahoma → cascadia (project rebrand) ([#31](https://github.com/labscommunity/cascadia/issues/31)) ([a1962a9](https://github.com/labscommunity/cascadia/commit/a1962a93cd5fb89be97df32d3b1b4249709593d0))
* **rust:** production security hardening (Phase 14) ([02b69e3](https://github.com/labscommunity/cascadia/commit/02b69e3bd63a29abd855ca066e2080de725e3f14))
* scaffold project structure ([a86d339](https://github.com/labscommunity/cascadia/commit/a86d339c3bdcf894f62987780f8160376bb89399))
* scrub internal references from code comments and CI ([a6b2062](https://github.com/labscommunity/cascadia/commit/a6b2062905fea8f6a6bc05829e92c0d9df5c4864))
* **shard:** catch r1-distill registry up to main ([297bcd8](https://github.com/labscommunity/cascadia/commit/297bcd85409516bc7c1b81c46c4d9cd40bd13b19))
* **shard:** scrub internal codename from gemma-4 text exporter comments ([50d8e77](https://github.com/labscommunity/cascadia/commit/50d8e77a5275ce11cd70d9c96f4da3656e1a424f))
* update readme image to cascadia ([3314887](https://github.com/labscommunity/cascadia/commit/3314887c10a7e0a18eb23e3c9e255db984c9e264))
