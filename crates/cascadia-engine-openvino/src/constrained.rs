//! llguidance-backed constrained decoding. Pure CPU (no OpenVINO), so it
//! unit-tests on any platform without `--features openvino`. See slice-3 spec.

use std::sync::Arc;

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::{GrammarKind, GrammarSpec};
use llguidance::{api::TopLevelGrammar, toktrie::TokEnv, Matcher, ParserFactory};
use toktrie_hf_tokenizers::ByteTokenizer;

fn backend<E: std::fmt::Display>(e: E) -> EngineError {
    EngineError::Backend(format!("constrained: {e}"))
}

/// Keep only EOS ids the tokenizer can represent (`set_eos_tokens` panics on
/// empty / out-of-range).
pub(crate) fn valid_eos(eos: &[u32], vocab: usize) -> Vec<u32> {
    eos.iter().copied().filter(|&t| (t as usize) < vocab).collect()
}

/// Tail-side: set every grammar-disallowed logit to -inf from a packed bitset.
/// A sentinel (len < vocab/8) means "unconstrained" -> no-op. Bit i = byte i/8, bit i%8.
pub fn apply_mask_bytes(logits_row: &mut [f32], mask: &[i8]) {
    let vocab = logits_row.len();
    if mask.len() * 8 < vocab {
        return; // sentinel / no constraint
    }
    for (i, l) in logits_row.iter_mut().enumerate() {
        let byte = mask[i / 8] as u8;
        if (byte >> (i % 8)) & 1 == 0 {
            *l = f32::NEG_INFINITY;
        }
    }
}

pub struct GrammarFactory {
    factory: Arc<ParserFactory>,
    mask_width: usize,
}

impl GrammarFactory {
    pub fn new(tokenizer: &tokenizers::Tokenizer, eos: &[u32]) -> EngineResult<Self> {
        let mut bt = ByteTokenizer::from_tokenizer(tokenizer.clone()).map_err(backend)?;
        let vocab = bt.tokrx_info().vocab_size as usize;
        let eos = valid_eos(eos, vocab);
        if !eos.is_empty() {
            bt.set_eos_tokens(&eos);
        }
        let tok_env = bt.into_tok_env(None).map_err(backend)?;
        Self::from_tok_env(tok_env, vocab)
    }

    fn from_tok_env(tok_env: TokEnv, mask_width: usize) -> EngineResult<Self> {
        let factory = Arc::new(ParserFactory::new_simple(&tok_env).map_err(backend)?);
        Ok(Self { factory, mask_width })
    }

    #[cfg(test)]
    fn single_byte() -> Self {
        let tok_env = llguidance::toktrie::ApproximateTokEnv::single_byte_env();
        let w = tok_env.tok_trie().vocab_size();
        Self::from_tok_env(tok_env, w).unwrap()
    }

    pub fn mask_width(&self) -> usize {
        self.mask_width
    }

    pub fn create(&self, grammar: &GrammarSpec) -> EngineResult<GrammarMask> {
        let top = match grammar.kind {
            GrammarKind::JsonSchema => {
                let v: serde_json::Value = serde_json::from_str(&grammar.body).map_err(backend)?;
                TopLevelGrammar::from_json_schema(v)
            }
        };
        Ok(GrammarMask {
            matcher: Matcher::new(self.factory.create_parser(top)),
            mask_width: self.mask_width,
        })
    }
}

pub enum ApplyOutcome {
    Masked,
    Complete,
}

pub struct GrammarMask {
    matcher: Matcher,
    mask_width: usize,
}

impl GrammarMask {
    pub fn apply(&mut self, row: &mut [f32]) -> EngineResult<ApplyOutcome> {
        if self.matcher.is_stopped() {
            return Ok(ApplyOutcome::Complete);
        }
        let mask = self.matcher.compute_mask_or_eos().map_err(backend)?;
        for (i, l) in row.iter_mut().enumerate() {
            if i >= self.mask_width || !mask.is_allowed(i as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        Ok(ApplyOutcome::Masked)
    }

    pub fn next_mask_bytes(&mut self) -> EngineResult<Vec<i8>> {
        let mask = self.matcher.compute_mask_or_eos().map_err(backend)?;
        let mut bytes = vec![0i8; self.mask_width.div_ceil(8)];
        for i in 0..self.mask_width {
            if mask.is_allowed(i as u32) {
                bytes[i / 8] |= (1u8 << (i % 8)) as i8;
            }
        }
        Ok(bytes)
    }

    pub fn accept(&mut self, token: u32) -> EngineResult<()> {
        self.matcher.consume_token(token).map_err(backend)?;
        Ok(())
    }

    pub fn is_stopped(&self) -> bool {
        self.matcher.is_stopped()
    }
}

/// 1-byte sentinel meaning "no constraint this step" (frame-count-stable wire).
pub fn sentinel_mask() -> Vec<i8> {
    vec![0i8]
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_types::{GrammarKind, GrammarSpec};

    fn hard_schema() -> GrammarSpec {
        GrammarSpec {
            kind: GrammarKind::JsonSchema,
            body: r#"{"type":"object","properties":{"city":{"type":"string"},"temperature_celsius":{"type":"number"},"conditions":{"type":"string","enum":["sunny","cloudy","rainy"]}},"required":["city","temperature_celsius","conditions"],"additionalProperties":false}"#.to_string(),
        }
    }

    fn validate(gf: &GrammarFactory, s: &str) -> Result<bool, usize> {
        let mut gm = gf.create(&hard_schema()).unwrap();
        for (i, &b) in s.as_bytes().iter().enumerate() {
            let mut row = vec![0.0f32; gf.mask_width()];
            match gm.apply(&mut row).unwrap() {
                ApplyOutcome::Complete => return Err(i),
                ApplyOutcome::Masked => {
                    if !row[b as usize].is_finite() { return Err(i); }
                    gm.accept(b as u32).unwrap();
                }
            }
        }
        Ok(gm.is_stopped())
    }

    #[test]
    fn accepts_conforming_json() {
        let gf = GrammarFactory::single_byte();
        assert_eq!(validate(&gf, r#"{"city":"Paris","temperature_celsius":18,"conditions":"sunny"}"#), Ok(true));
    }

    #[test]
    fn rejects_enum_extra_and_missing() {
        let gf = GrammarFactory::single_byte();
        assert!(validate(&gf, r#"{"city":"Paris","temperature_celsius":18,"conditions":"stormy"}"#).is_err());
        assert!(validate(&gf, r#"{"city":"Paris","temperature_celsius":18,"conditions":"sunny","x":1}"#).is_err());
        assert!(validate(&gf, r#"{"city":"Paris","conditions":"sunny"}"#).is_err());
    }

    #[test]
    fn first_masked_token_is_open_brace() {
        let gf = GrammarFactory::single_byte();
        let mut gm = gf.create(&hard_schema()).unwrap();
        let mut row = vec![0.0f32; gf.mask_width()];
        assert!(matches!(gm.apply(&mut row).unwrap(), ApplyOutcome::Masked));
        assert!(row[b'{' as usize].is_finite());
        assert_eq!(row[b'x' as usize], f32::NEG_INFINITY);
    }

    #[test]
    fn valid_eos_filters_out_of_range_and_keeps_empty_empty() {
        assert_eq!(valid_eos(&[1, 9_999_999, 2], 128_256), vec![1, 2]);
        assert!(valid_eos(&[], 128_256).is_empty());
        assert!(valid_eos(&[128_256], 128_256).is_empty());
    }

    #[test]
    fn wire_mask_roundtrip_matches_local_apply() {
        let gf = GrammarFactory::single_byte();
        let mut local = gf.create(&hard_schema()).unwrap();
        let mut row_local = vec![0.0f32; gf.mask_width()];
        local.apply(&mut row_local).unwrap();
        let mut head = gf.create(&hard_schema()).unwrap();
        let bytes = head.next_mask_bytes().unwrap();
        let mut row_wire = vec![0.0f32; gf.mask_width()];
        apply_mask_bytes(&mut row_wire, &bytes);
        assert_eq!(row_local, row_wire);
        let mut row_sentinel = vec![0.0f32; gf.mask_width()];
        apply_mask_bytes(&mut row_sentinel, &[0i8]);
        assert!(row_sentinel.iter().all(|v| v.is_finite()));
    }
}
