// Compile-only skeleton; Task 6 replaces this file.
use super::{
    profile::KevProfile,
    schema::{DecisionRequest, Limits},
    Codec, DecisionError, Pass, PassPart,
};
pub struct KevCodec;
impl KevCodec {
    pub fn new(_tok: tokenizers::Tokenizer, _p: KevProfile) -> Result<Self, String> {
        Err("kev codec not built yet".into())
    }
}
impl Codec for KevCodec {
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
