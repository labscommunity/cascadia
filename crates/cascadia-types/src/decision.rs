//! Decision readout: a prefill-only pass whose result is per-option logits
//! computed by the model's decision head (see docs/DECISIONS.md). The API's
//! codec tokenizes and lays out the prompt; the engine prefills `prompt_ids`
//! exactly (no chat template, no think block, no special tokens added) and
//! never decodes.

use serde::{Deserialize, Serialize};

/// Upper bound on logits one readout may request (Clef: options summed over questions).
pub const MAX_DECISION_OUTPUTS: usize = 4096;
/// Upper bound on questions in one joint (Clef) readout.
pub const MAX_JOINT_QUESTIONS: usize = 256;
/// Prompt-length bound for a decision pass when the engine does not know the model window
/// (pipeline ranks other than 0 never read `max_position_embeddings`).
pub const MAX_DECISION_PROMPT: usize = 131_072;
/// Prompt-length bound for a joint (Clef) readout, Clef's training `max_length`. The effective
/// bound also depends on the model width: see [`joint_prompt_cap`] (16,384 for both Clef-flash's
/// 4096 and Clef-27B's 5120; only a wider tree is tightened).
///
/// Memory: the joint head reads EVERY final-hidden row, and `run_head` hands OpenVINO a dense
/// `[R, T]` f32 span mask (R = questions + options). Per pass the head-owning process holds the
/// collected rows (`T * hidden * 4` bytes), OpenVINO's copy of them, and the mask (`R * T * 4`)
/// plus OpenVINO's copy; the mask buffer is freed before the rows are copied, so the transient
/// peak is `rows + max(2 * mask, mask + rows)`. [`joint_inputs_fit`] holds every input to
/// [`MAX_HEAD_INPUT_BYTES`], so that peak is at most 3 x 320 MiB (~0.94 GiB), checked before any
/// allocation or prefill, for HTTP and peer readouts alike. OpenVINO keeps its input copies
/// until the next joint pass.
pub const MAX_JOINT_PROMPT: usize = 16_384;

/// Largest input tensor the native shim accepts: `MAX_TENSOR_BYTES` in
/// crates/cascadia-ov-genai-shim/cpp/shim.cpp (`cascadia_runtime_set_input` refuses anything
/// larger). Mirrored here so a joint readout that would overflow it is refused up front, as a
/// too-long request, instead of failing inside the shim after a full prefill; a test parses the
/// C++ constant to keep the two equal. Sized to Clef-27B's joint `hidden` input at
/// [`MAX_JOINT_PROMPT`]: [16384, 5120] f32 is exactly 320 MiB, with no slack (each input is
/// checked on its own, and the others are smaller: lexical <= 80 MiB, span mask <= 272 MiB).
pub const MAX_HEAD_INPUT_BYTES: usize = 320 * 1024 * 1024;

/// Shortest prefix the engine's prefix cache stores: a snapshot under it is not worth the
/// `get_state_blob` copy. Shared so the API never counts a reuse discount the cache cannot give.
pub const MIN_PREFIX_TOKENS: usize = 16;

/// The joint prompt bound at model width `hidden`: [`MAX_JOINT_PROMPT`], tightened so the head's
/// `hidden` input (`[T, hidden]` f32) fits [`MAX_HEAD_INPUT_BYTES`]. 4096 wide (Clef-flash) and
/// 5120 wide (Clef-27B): 16,384; a wider tree is tightened (8192 wide: 10,240).
pub fn joint_prompt_cap(hidden: usize) -> usize {
    MAX_JOINT_PROMPT.min(MAX_HEAD_INPUT_BYTES / (4 * hidden.max(1)))
}

/// Longest joint prompt whose `[rows, T]` f32 span mask fits [`MAX_HEAD_INPUT_BYTES`]. At most
/// 256 questions + 4096 options give 19,275, so today it never binds below [`MAX_JOINT_PROMPT`].
pub fn joint_mask_prompt_cap(rows: usize) -> usize {
    MAX_HEAD_INPUT_BYTES / (4 * rows.max(1))
}

/// Every f32 input of a joint head pass fits the shim's [`MAX_HEAD_INPUT_BYTES`]:
/// `hidden` `[t, hidden]`, `span_mask` `[rows, t]` (rows = questions + options) and `lexical`
/// `[options, hidden]`. `Err` names the input and the limit.
pub fn joint_inputs_fit(
    t: usize,
    rows: usize,
    options: usize,
    hidden: usize,
) -> Result<(), String> {
    if t > joint_prompt_cap(hidden) {
        return Err(format!(
            "joint readout of {t} tokens exceeds {} at width {hidden}",
            joint_prompt_cap(hidden)
        ));
    }
    if t > joint_mask_prompt_cap(rows) {
        return Err(format!(
            "joint readout of {t} tokens x {rows} spans exceeds the {MAX_HEAD_INPUT_BYTES}-byte span mask limit"
        ));
    }
    if options.saturating_mul(hidden).saturating_mul(4) > MAX_HEAD_INPUT_BYTES {
        return Err(format!(
            "joint readout of {options} options at width {hidden} exceeds the {MAX_HEAD_INPUT_BYTES}-byte lexical limit"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionReadout {
    /// Exact ids to prefill, already rendered and tokenized by the codec.
    pub prompt_ids: Vec<u32>,
    pub head: ReadoutHead,
    /// `prompt_ids[..cache_prefix]` is shared by every pass of one request (Kev: the state).
    /// The engine snapshots the prefix cache there so later passes prefill only their tail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_prefix: Option<u32>,
    /// The pass must resume from the prefix cache at `cache_prefix` or later (single box). Set by
    /// the API on a later pass of a request whose budget counted the shared prefix once: if the
    /// snapshot was evicted, the engine refuses the pass (`decision_cold`) instead of prefilling
    /// the prefix again past the budget the request was admitted under.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub require_warm: bool,
    /// The first pass of a request whose later passes are `require_warm` (single box): the engine
    /// checks, BEFORE prefilling it, that the prefix cache can hold a snapshot at `cache_prefix`
    /// at all (`decision_uncacheable`, a 413, when it cannot), and then always takes that
    /// snapshot, whatever the chat refresh heuristics would decide.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub needs_snapshot: bool,
}

/// Which head reads the prefilled state. Positions/spans index `prompt_ids`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReadoutHead {
    /// JEV: lm_head logits at the last position for these vocab ids.
    Verbalizer { token_ids: Vec<u32> },
    /// Kev: pointer head over final-norm hidden states at `option_pos`
    /// (each option's closing token) against `decide_pos`.
    Pointer {
        decide_pos: u32,
        option_pos: Vec<u32>,
    },
    /// Clef: joint schema head over all final-norm hidden states.
    Joint { questions: Vec<JointQuestion> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointQuestion {
    /// 0 = noul, 1 = choice, 2 = score (Clef's QUESTION_TYPES).
    pub qtype: u8,
    /// Half-open [start, end) token span of the instruction text.
    pub span: [u32; 2],
    pub options: Vec<JointOption>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointOption {
    /// Half-open [start, end) token span of the option's rendered semantics.
    pub span: [u32; 2],
}

/// Final-norm hidden rows the head needs collected during prefill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HiddenNeed {
    None,
    /// Sorted, de-duplicated absolute positions.
    Positions(Vec<u32>),
    All,
}

impl DecisionReadout {
    /// Bounds-check everything an engine indexes with. Readouts cross a trust boundary (HTTP
    /// clients via the codec, and pipeline peers via FRAME_DECIDE), so the engine calls this on
    /// every admission instead of trusting the sender.
    pub fn validate(&self, vocab: usize, max_tokens: usize) -> Result<(), String> {
        let n = self.prompt_ids.len();
        if n == 0 || n > max_tokens {
            return Err(format!(
                "decision prompt has {n} tokens; allowed 1..={max_tokens}"
            ));
        }
        if let Some(&bad) = self.prompt_ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!("decision prompt id {bad} >= vocab {vocab}"));
        }
        if let Some(c) = self.cache_prefix {
            if c as usize >= n {
                return Err(format!("cache_prefix {c} must be < prompt length {n}"));
            }
        }
        let out = self.head.output_len();
        if out == 0 || out > MAX_DECISION_OUTPUTS {
            return Err(format!(
                "decision head asks for {out} logits; allowed 1..={MAX_DECISION_OUTPUTS}"
            ));
        }
        let span_ok = |s: &[u32; 2]| s[0] < s[1] && s[1] as usize <= n;
        match &self.head {
            ReadoutHead::Verbalizer { token_ids } => {
                if let Some(&bad) = token_ids.iter().find(|&&t| t as usize >= vocab) {
                    return Err(format!("verbalizer id {bad} >= vocab {vocab}"));
                }
            }
            ReadoutHead::Pointer {
                decide_pos,
                option_pos,
            } => {
                if let Some(&bad) = option_pos
                    .iter()
                    .chain(std::iter::once(decide_pos))
                    .find(|&&p| p as usize >= n)
                {
                    return Err(format!(
                        "pointer position {bad} outside the {n}-token prompt"
                    ));
                }
            }
            ReadoutHead::Joint { questions } => {
                if n > MAX_JOINT_PROMPT {
                    return Err(format!(
                        "joint readout of {n} tokens exceeds {MAX_JOINT_PROMPT}"
                    ));
                }
                if questions.is_empty() || questions.len() > MAX_JOINT_QUESTIONS {
                    return Err(format!(
                        "joint readout has {} questions; allowed 1..={MAX_JOINT_QUESTIONS}",
                        questions.len()
                    ));
                }
                // Spans must be in prompt order and disjoint (question span, its options, next question…),
                // exactly as the codec lays them out. Overlapping spans would let a peer multiply
                // lm_head reads and span-mask size without bound.
                let mut cursor = 0u32;
                for (i, q) in questions.iter().enumerate() {
                    if q.qtype > 2 || q.options.is_empty() {
                        return Err(format!("joint question {i}: bad qtype or no options"));
                    }
                    for s in std::iter::once(&q.span).chain(q.options.iter().map(|o| &o.span)) {
                        if !span_ok(s) || s[0] < cursor {
                            return Err(format!("joint question {i}: span {s:?} empty, out of range or out of order"));
                        }
                        cursor = s[1];
                    }
                }
            }
        }
        Ok(())
    }

    /// Longest prefix the engine may restore from its prefix cache for this pass. A restored
    /// prefix is never re-run, so no hidden row or logit inside it is produced: the limit is
    /// the first position the head reads (Verbalizer: the last token's logits; Joint: every row).
    pub fn warm_limit(&self) -> usize {
        match &self.head {
            ReadoutHead::Verbalizer { .. } => self.prompt_ids.len().saturating_sub(1),
            ReadoutHead::Pointer {
                decide_pos,
                option_pos,
            } => option_pos
                .iter()
                .chain(std::iter::once(decide_pos))
                .copied()
                .min()
                .unwrap_or(0) as usize,
            ReadoutHead::Joint { .. } => 0,
        }
    }
}

impl ReadoutHead {
    /// Number of logits the engine returns for this head.
    pub fn output_len(&self) -> usize {
        match self {
            ReadoutHead::Verbalizer { token_ids } => token_ids.len(),
            ReadoutHead::Pointer { option_pos, .. } => option_pos.len(),
            ReadoutHead::Joint { questions } => questions.iter().map(|q| q.options.len()).sum(),
        }
    }

    pub fn hidden_need(&self) -> HiddenNeed {
        match self {
            ReadoutHead::Verbalizer { .. } => HiddenNeed::None,
            ReadoutHead::Pointer {
                decide_pos,
                option_pos,
            } => {
                let mut v = option_pos.clone();
                v.push(*decide_pos);
                v.sort_unstable();
                v.dedup();
                HiddenNeed::Positions(v)
            }
            ReadoutHead::Joint { .. } => HiddenNeed::All,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_prompt_cap_follows_the_shim_input_limit() {
        // Clef-27B (5120 wide): [16384, 5120] f32 is exactly 320 MiB, the shim's limit, so the
        // training max_length binds for both Clef models.
        assert_eq!(MAX_JOINT_PROMPT * 5120 * 4, MAX_HEAD_INPUT_BYTES);
        assert_eq!(joint_prompt_cap(5120), MAX_JOINT_PROMPT);
        assert_eq!(joint_prompt_cap(4096), MAX_JOINT_PROMPT);
        assert_eq!(joint_prompt_cap(0), MAX_JOINT_PROMPT);
        // The bound still holds a wider model to the limit: 2^28 * 1.25 / (4 * 8192) = 10240.
        assert_eq!(joint_prompt_cap(8192), 10_240);
        assert_eq!(joint_prompt_cap(6144), 13_653);
    }

    #[test]
    fn head_input_limit_mirrors_the_shim() {
        // MAX_HEAD_INPUT_BYTES mirrors MAX_TENSOR_BYTES in shim.cpp (compiled into the same
        // binary by build.rs): parse the C++ constant so the two can never drift apart.
        let src = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cascadia-ov-genai-shim/cpp/shim.cpp"
        ));
        let line = src
            .lines()
            .find(|l| l.contains("constexpr size_t MAX_TENSOR_BYTES ="))
            .expect("shim.cpp declares MAX_TENSOR_BYTES");
        let expr = line.split('=').nth(1).unwrap().split(';').next().unwrap();
        let bytes: usize = expr
            .split('*')
            .map(|f| f.trim().parse::<usize>().expect("integer factor"))
            .product();
        assert_eq!(bytes, MAX_HEAD_INPUT_BYTES, "{line}");
        let mib = format!("MAX_TENSOR_BYTES={}MiB", MAX_HEAD_INPUT_BYTES >> 20);
        assert!(src.contains(&mib), "shim error text must say {mib}");
        // The margin is zero (Clef-27B's `hidden` at MAX_JOINT_PROMPT is exactly the limit), so
        // the shim must refuse only what is strictly larger: `>=` would refuse an admitted input.
        assert!(
            src.contains("if (data_size > MAX_TENSOR_BYTES)"),
            "shim must compare with a strict >"
        );
    }

    #[test]
    fn joint_mask_bound_is_rows_times_t() {
        // At most 256 questions + 4096 options = 4352 spans: 320 MiB / (4 * 4352) = 19275, so
        // the mask never binds below MAX_JOINT_PROMPT; it still bounds a larger row count.
        assert_eq!(joint_mask_prompt_cap(4352), 19_275);
        assert_eq!(joint_mask_prompt_cap(8192), 10_240);
        assert!(joint_inputs_fit(10_240, 8192, 4096, 4096).is_ok());
        let e = joint_inputs_fit(10_241, 8192, 4096, 4096).unwrap_err();
        assert!(e.contains("span mask"), "{e}");
        assert!(joint_inputs_fit(MAX_JOINT_PROMPT, 4352, 4096, 5120).is_ok());
        let e = joint_inputs_fit(MAX_JOINT_PROMPT + 1, 2, 1, 5120).unwrap_err();
        assert!(e.contains("width 5120"), "{e}");
        let e = joint_inputs_fit(10_241, 2, 1, 8192).unwrap_err();
        assert!(e.contains("width 8192"), "{e}");
    }

    #[test]
    fn verbalizer_round_trips_and_sizes() {
        let r = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Verbalizer {
                token_ids: vec![10, 11],
            },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"kind\":\"verbalizer\""));
        let back: DecisionReadout = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
        assert_eq!(r.head.output_len(), 2);
        assert_eq!(r.head.hidden_need(), HiddenNeed::None);
        // The reuse flags stay off the wire when false (pipeline frames are unchanged) and
        // default to false from an older peer's readout.
        assert!(
            !s.contains("require_warm") && !s.contains("needs_snapshot"),
            "{s}"
        );
        let flagged = DecisionReadout {
            needs_snapshot: true,
            ..r.clone()
        };
        let s = serde_json::to_string(&flagged).unwrap();
        assert!(s.contains("\"needs_snapshot\":true"), "{s}");
        assert_eq!(
            serde_json::from_str::<DecisionReadout>(&s).unwrap(),
            flagged
        );
    }

    #[test]
    fn pointer_needs_its_positions() {
        let h = ReadoutHead::Pointer {
            decide_pos: 9,
            option_pos: vec![4, 7],
        };
        assert_eq!(h.output_len(), 2);
        assert_eq!(h.hidden_need(), HiddenNeed::Positions(vec![4, 7, 9]));
    }

    #[test]
    fn joint_needs_all_rows_and_counts_options() {
        let h = ReadoutHead::Joint {
            questions: vec![
                JointQuestion {
                    qtype: 1,
                    span: [2, 4],
                    options: vec![JointOption { span: [5, 6] }, JointOption { span: [7, 9] }],
                },
                JointQuestion {
                    qtype: 0,
                    span: [10, 11],
                    options: vec![JointOption { span: [12, 13] }],
                },
            ],
        };
        assert_eq!(h.output_len(), 3);
        assert_eq!(h.hidden_need(), HiddenNeed::All);
    }

    #[test]
    fn validate_caps_joint_readouts_at_max_joint_prompt() {
        let joint = |n: u32| DecisionReadout {
            prompt_ids: (0..n).collect(),
            head: ReadoutHead::Joint {
                questions: vec![JointQuestion {
                    qtype: 0,
                    span: [0, 1],
                    options: vec![JointOption { span: [1, 2] }],
                }],
            },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        let vocab = MAX_JOINT_PROMPT + 10;
        assert!(joint(MAX_JOINT_PROMPT as u32)
            .validate(vocab, MAX_DECISION_PROMPT)
            .is_ok());
        let err = joint(MAX_JOINT_PROMPT as u32 + 1)
            .validate(vocab, MAX_DECISION_PROMPT)
            .unwrap_err();
        assert_eq!(err, "joint readout of 16385 tokens exceeds 16384");
        // Non-joint heads keep the window bound only.
        let long_ptr = DecisionReadout {
            prompt_ids: (0..MAX_JOINT_PROMPT as u32 + 1).collect(),
            head: ReadoutHead::Pointer {
                decide_pos: 3,
                option_pos: vec![1],
            },
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        };
        assert!(long_ptr.validate(vocab, MAX_DECISION_PROMPT).is_ok());
    }

    fn readout(prompt_len: u32, head: ReadoutHead) -> DecisionReadout {
        DecisionReadout {
            prompt_ids: (0..prompt_len).collect(),
            head,
            cache_prefix: None,
            require_warm: false,
            needs_snapshot: false,
        }
    }

    #[test]
    fn validate_rejects_out_of_range_ids_positions_and_spans() {
        let ok = readout(
            10,
            ReadoutHead::Pointer {
                decide_pos: 9,
                option_pos: vec![3, 5],
            },
        );
        assert!(ok.validate(100, 64).is_ok());
        assert!(
            readout(
                10,
                ReadoutHead::Verbalizer {
                    token_ids: vec![100]
                }
            )
            .validate(100, 64)
            .is_err(),
            "id >= vocab"
        );
        assert!(
            readout(0, ReadoutHead::Verbalizer { token_ids: vec![1] })
                .validate(100, 64)
                .is_err(),
            "empty prompt"
        );
        assert!(
            readout(65, ReadoutHead::Verbalizer { token_ids: vec![1] })
                .validate(100, 64)
                .is_err(),
            "prompt > max_tokens"
        );
        let mut bad_id = readout(3, ReadoutHead::Verbalizer { token_ids: vec![1] });
        bad_id.prompt_ids[1] = 100;
        assert!(bad_id.validate(100, 64).is_err(), "prompt id >= vocab");
        assert!(readout(
            10,
            ReadoutHead::Pointer {
                decide_pos: 10,
                option_pos: vec![3]
            }
        )
        .validate(100, 64)
        .is_err());
        let span = |a: u32, b: u32| ReadoutHead::Joint {
            questions: vec![JointQuestion {
                qtype: 0,
                span: [a, b],
                options: vec![JointOption { span: [1, 2] }],
            }],
        };
        assert!(
            readout(10, span(4, 4)).validate(100, 64).is_err(),
            "empty span"
        );
        assert!(
            readout(10, span(5, 4)).validate(100, 64).is_err(),
            "reversed span"
        );
        assert!(
            readout(10, span(4, 11)).validate(100, 64).is_err(),
            "span past prompt"
        );
        assert!(
            readout(10, ReadoutHead::Joint { questions: vec![] })
                .validate(100, 64)
                .is_err(),
            "no questions"
        );
        let overlap = ReadoutHead::Joint {
            questions: vec![JointQuestion {
                qtype: 1,
                span: [1, 2],
                options: vec![JointOption { span: [3, 6] }, JointOption { span: [4, 7] }],
            }],
        };
        assert!(
            readout(10, overlap).validate(100, 64).is_err(),
            "overlapping option spans"
        );
        assert!(
            readout(
                10,
                ReadoutHead::Joint {
                    questions: vec![JointQuestion {
                        qtype: 3,
                        span: [1, 2],
                        options: vec![JointOption { span: [2, 3] }]
                    }]
                }
            )
            .validate(100, 64)
            .is_err(),
            "bad qtype"
        );
        let mut cp = readout(10, ReadoutHead::Verbalizer { token_ids: vec![1] });
        cp.cache_prefix = Some(10);
        assert!(
            cp.validate(100, 64).is_err(),
            "cache_prefix must leave >= 1 token to prefill"
        );
    }

    #[test]
    fn validate_pins_every_reason() {
        let joint = |questions: Vec<JointQuestion>| ReadoutHead::Joint { questions };
        let q = |qtype: u8, span: [u32; 2], opts: &[[u32; 2]]| JointQuestion {
            qtype,
            span,
            options: opts.iter().map(|&span| JointOption { span }).collect(),
        };
        let err = |r: DecisionReadout| r.validate(100, 64).unwrap_err();
        let verb = |ids: Vec<u32>| ReadoutHead::Verbalizer { token_ids: ids };
        assert_eq!(
            err(readout(0, verb(vec![1]))),
            "decision prompt has 0 tokens; allowed 1..=64"
        );
        assert_eq!(
            err(readout(65, verb(vec![1]))),
            "decision prompt has 65 tokens; allowed 1..=64"
        );
        let mut bad_id = readout(3, verb(vec![1]));
        bad_id.prompt_ids[1] = 100;
        assert_eq!(err(bad_id), "decision prompt id 100 >= vocab 100");
        assert_eq!(
            err(readout(10, verb(vec![100]))),
            "verbalizer id 100 >= vocab 100"
        );
        assert_eq!(
            err(readout(10, verb(vec![]))),
            format!("decision head asks for 0 logits; allowed 1..={MAX_DECISION_OUTPUTS}")
        );
        // One past MAX_DECISION_OUTPUTS (ids are in range, so only the count is wrong).
        assert_eq!(
            readout(10, verb(vec![1; MAX_DECISION_OUTPUTS + 1]))
                .validate(100, 64)
                .unwrap_err(),
            format!(
                "decision head asks for {} logits; allowed 1..={MAX_DECISION_OUTPUTS}",
                MAX_DECISION_OUTPUTS + 1
            )
        );
        assert!(readout(10, verb(vec![1; MAX_DECISION_OUTPUTS]))
            .validate(100, 64)
            .is_ok());
        assert_eq!(
            err(readout(
                10,
                ReadoutHead::Pointer {
                    decide_pos: 10,
                    option_pos: vec![3]
                }
            )),
            "pointer position 10 outside the 10-token prompt"
        );
        let mut cp = readout(10, verb(vec![1]));
        cp.cache_prefix = Some(10);
        assert_eq!(err(cp), "cache_prefix 10 must be < prompt length 10");
        // A valid cache_prefix (the last one that leaves a token to prefill) is accepted.
        let mut cp = readout(10, verb(vec![1]));
        cp.cache_prefix = Some(9);
        assert!(cp.validate(100, 64).is_ok());
        assert_eq!(
            err(readout(10, joint(vec![]))),
            format!("decision head asks for 0 logits; allowed 1..={MAX_DECISION_OUTPUTS}")
        );
        // More than MAX_JOINT_QUESTIONS questions (one option each, so the logit count is fine).
        let n = 2 * (MAX_JOINT_QUESTIONS as u32 + 1) + 1;
        let many: Vec<_> = (0..=MAX_JOINT_QUESTIONS as u32)
            .map(|i| q(0, [2 * i, 2 * i + 1], &[[2 * i + 1, 2 * i + 2]]))
            .collect();
        assert_eq!(
            readout(n, joint(many.clone()))
                .validate(1000, 1000)
                .unwrap_err(),
            format!(
                "joint readout has {} questions; allowed 1..={MAX_JOINT_QUESTIONS}",
                MAX_JOINT_QUESTIONS + 1
            )
        );
        assert!(readout(n, joint(many[1..].to_vec()))
            .validate(1000, 1000)
            .is_ok());
        assert_eq!(
            err(readout(10, joint(vec![q(3, [1, 2], &[[2, 3]])]))),
            "joint question 0: bad qtype or no options"
        );
        assert_eq!(
            err(readout(10, joint(vec![q(0, [4, 4], &[[5, 6]])]))),
            "joint question 0: span [4, 4] empty, out of range or out of order"
        );
        // Each question is in order on its own, but the second starts before the first ends.
        assert_eq!(
            err(readout(
                10,
                joint(vec![q(0, [4, 5], &[[5, 6]]), q(0, [1, 2], &[[2, 3]])])
            )),
            "joint question 1: span [1, 2] empty, out of range or out of order"
        );
        assert!(readout(
            10,
            joint(vec![
                q(0, [1, 2], &[[2, 3]]),
                q(1, [4, 5], &[[5, 6], [6, 8]])
            ])
        )
        .validate(100, 64)
        .is_ok());
    }

    #[test]
    fn warm_limit_keeps_every_row_the_head_reads() {
        assert_eq!(
            readout(10, ReadoutHead::Verbalizer { token_ids: vec![1] }).warm_limit(),
            9
        );
        assert_eq!(
            readout(
                10,
                ReadoutHead::Pointer {
                    decide_pos: 9,
                    option_pos: vec![6, 4]
                }
            )
            .warm_limit(),
            4
        );
        let j = ReadoutHead::Joint {
            questions: vec![JointQuestion {
                qtype: 0,
                span: [1, 2],
                options: vec![JointOption { span: [2, 3] }],
            }],
        };
        assert_eq!(readout(10, j).warm_limit(), 0);
    }

    #[test]
    fn warm_limit_does_not_underflow_on_an_empty_prompt() {
        // Callers validate first today; warm_limit must not panic if one ever does not.
        assert_eq!(
            readout(0, ReadoutHead::Verbalizer { token_ids: vec![1] }).warm_limit(),
            0
        );
    }

    #[test]
    fn validate_rejects_empty_option_lists_for_every_head() {
        // Empty options would softmax over nothing (NaN); a peer must not be able to send them.
        assert!(
            readout(10, ReadoutHead::Verbalizer { token_ids: vec![] })
                .validate(100, 64)
                .is_err(),
            "verbalizer with no ids"
        );
        assert!(
            readout(
                10,
                ReadoutHead::Pointer {
                    decide_pos: 9,
                    option_pos: vec![]
                }
            )
            .validate(100, 64)
            .is_err(),
            "pointer with no options"
        );
        // A joint readout whose total is non-zero but one question has no options.
        let joint = ReadoutHead::Joint {
            questions: vec![
                JointQuestion {
                    qtype: 1,
                    span: [1, 2],
                    options: vec![JointOption { span: [2, 3] }],
                },
                JointQuestion {
                    qtype: 1,
                    span: [4, 5],
                    options: vec![],
                },
            ],
        };
        assert_eq!(
            readout(10, joint).validate(100, 64).unwrap_err(),
            "joint question 1: bad qtype or no options"
        );
    }

    #[test]
    fn task_without_decision_field_still_parses() {
        let t: crate::GenerationTask =
            serde_json::from_str(r#"{"task_id":"t","prompt":"p"}"#).unwrap();
        assert!(t.decision.is_none());
    }

    #[test]
    fn chunk_carries_decision_logits() {
        let c = crate::Chunk::final_marker("t", "").with_decision_logits(vec![0.5, -1.0]);
        let s = serde_json::to_string(&c).unwrap();
        let back: crate::Chunk = serde_json::from_str(&s).unwrap();
        assert_eq!(back.decision_logits, Some(vec![0.5, -1.0]));
        let plain = serde_json::to_string(&crate::Chunk::final_marker("t", "")).unwrap();
        assert!(!plain.contains("decision_logits"));
    }
}
