//! Little-endian tensors in a bounded frame. Header length is big endian.
//! Transport admission and encryption are mandatory responsibilities of callers.
use crate::{
    lease::LeaseId,
    manifest::{digest, identifier, invalid, ports_bytes, Port, MAX_TENSOR_BYTES},
    Result,
};
use serde::{Deserialize, Serialize};
pub const MAX_HEADER: usize = 64 * 1024;
pub const MAX_FRAME: usize = 8 + MAX_HEADER + MAX_TENSOR_BYTES;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub manifest: String,
    pub component: String,
    pub lease: LeaseId,
    pub sequence: u32,
    pub tensors: Vec<Port>,
}
impl Header {
    pub fn validate(&self) -> Result<usize> {
        digest(&self.manifest)?;
        identifier(&self.component)?;
        if self.sequence == 0 {
            return invalid("zero call sequence");
        }
        ports_bytes(&self.tensors)
    }
}
pub fn encode(header: &Header, values: &[&[u8]]) -> Result<Vec<u8>> {
    let bytes = header.validate()?;
    if values.len() != header.tensors.len() {
        return invalid("tensor count mismatch");
    }
    let json = serde_json::to_vec(header)?;
    if json.len() > MAX_HEADER {
        return invalid("header exceeds limit");
    }
    let mut out = Vec::with_capacity(8 + json.len() + bytes);
    out.extend_from_slice(b"CDF1");
    out.extend_from_slice(&(json.len() as u32).to_be_bytes());
    out.extend_from_slice(&json);
    for (port, value) in header.tensors.iter().zip(values) {
        if value.len() != port.bytes()? || !port.dtype.finite(value) {
            return invalid("invalid tensor bytes or nonfinite value");
        }
        out.extend_from_slice(value);
    }
    Ok(out)
}
/// Validates the entire frame before returning borrowed payload slices.
pub fn decode(frame: &[u8]) -> Result<(Header, Vec<&[u8]>)> {
    if frame.len() < 8 || frame.len() > MAX_FRAME || &frame[..4] != b"CDF1" {
        return invalid("invalid frame prefix or size");
    }
    let n = u32::from_be_bytes(frame[4..8].try_into().unwrap()) as usize;
    if n == 0 || n > MAX_HEADER || frame.len() < 8 + n {
        return invalid("invalid header length");
    }
    let header: Header = serde_json::from_slice(&frame[8..8 + n])?;
    let bytes = header.validate()?;
    if frame.len() != 8 + n + bytes {
        return invalid("payload length mismatch");
    }
    let mut offset = 8 + n;
    let mut values = Vec::with_capacity(header.tensors.len());
    for port in &header.tensors {
        let end = offset + port.bytes()?;
        let value = &frame[offset..end];
        if !port.dtype.finite(value) {
            return invalid("nonfinite tensor");
        }
        values.push(value);
        offset = end;
    }
    Ok((header, values))
}
/// Input/output order is part of the pinned contract, avoiding alias ambiguity.
pub fn expect(header: &Header, expected: &[Port]) -> Result<()> {
    if header.tensors != expected {
        return invalid("tensor schema mismatch");
    }
    Ok(())
}
