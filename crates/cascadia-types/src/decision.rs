//! Decision readout: a prefill-only pass whose result is per-option logits
//! computed by the model's decision head (see docs/decisions.md). The API's
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
/// Prompt-length bound for a joint (Clef) readout, Clef's training `max_length`. The joint head
/// reads EVERY final-hidden row, so the head-owning process holds `T * hidden * 4` bytes for the
/// pass (e.g. 131072 * 5120 * 4 = 2.7 GB at the window bound); capping T here bounds that at
/// ~335 MB for a 5120-wide model, before any allocation or prefill, for HTTP and peer readouts alike.
pub const MAX_JOINT_PROMPT: usize = 16_384;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DecisionReadout {
    /// Exact ids to prefill, already rendered and tokenized by the codec.
    pub prompt_ids: Vec<u32>,
    pub head: ReadoutHead,
    /// `prompt_ids[..cache_prefix]` is shared by every pass of one request (Kev: the state).
    /// The engine snapshots the prefix cache there so later passes prefill only their tail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_prefix: Option<u32>,
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
            ReadoutHead::Verbalizer { .. } => self.prompt_ids.len() - 1,
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
    fn verbalizer_round_trips_and_sizes() {
        let r = DecisionReadout {
            prompt_ids: vec![1, 2, 3],
            head: ReadoutHead::Verbalizer {
                token_ids: vec![10, 11],
            },
            cache_prefix: None,
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"kind\":\"verbalizer\""));
        let back: DecisionReadout = serde_json::from_str(&s).unwrap();
        assert_eq!(back, r);
        assert_eq!(r.head.output_len(), 2);
        assert_eq!(r.head.hidden_need(), HiddenNeed::None);
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
        };
        assert!(long_ptr.validate(vocab, MAX_DECISION_PROMPT).is_ok());
    }

    fn readout(prompt_len: u32, head: ReadoutHead) -> DecisionReadout {
        DecisionReadout {
            prompt_ids: (0..prompt_len).collect(),
            head,
            cache_prefix: None,
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
