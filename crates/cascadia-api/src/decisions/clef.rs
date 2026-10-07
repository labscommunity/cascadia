// Compile-only skeleton; Task 7 replaces this file.
use super::{
    schema::{DecisionRequest, Limits},
    Codec, DecisionError, Pass, PassPart,
};
pub struct ClefCodec;
impl ClefCodec {
    pub fn new(_tok: tokenizers::Tokenizer, _max_tokens: usize) -> Result<Self, String> {
        Err("clef codec not built yet".into())
    }
}
impl Codec for ClefCodec {
    fn limits(&self) -> Limits {
        Limits {
            max_questions: 0,
            max_options: 0,
            min_levels: 0,
            max_levels: 0,
        }
    }
    fn encode(&self, _r: &DecisionRequest) -> Result<Vec<Pass>, DecisionError> {
        Err(DecisionError::NotServed)
    }
    fn probabilities(&self, _r: &DecisionRequest, _p: &PassPart, _l: &[f32]) -> Vec<f64> {
        Vec::new()
    }
}
