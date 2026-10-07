// Compile-only skeleton; Task 5 replaces this file.
use super::{
    profile::JevProfile,
    schema::{DecisionRequest, Limits},
    Codec, DecisionError, Pass, PassPart,
};
pub struct JevCodec;
impl JevCodec {
    pub fn new(_tok: tokenizers::Tokenizer, _p: JevProfile) -> Result<Self, String> {
        Err("jev codec not built yet".into())
    }
}
impl Codec for JevCodec {
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
