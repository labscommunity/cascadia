//! dsv4 rotary embeddings: YaRN-scaled frequency table + the reference's
//! pair-interleaved complex rotation (with the inverse/conjugate variant V4
//! applies to attention *outputs*).
//!
//! Ports `precompute_freqs_cis` / `apply_rotary_emb` from
//! `tools/deepseek_v4_ref/model.py`. Pairs are adjacent (even, odd) elements
//! of the last dim — NOT rotate-half. V4 uses two tables: plain rope_theta on
//! pure-sliding-window layers, compress_rope_theta (+ YaRN when
//! original_seq_len > 0) on compressed layers.

use super::math::to_bf16;

/// cos/sin table: `[seq_len][dim/2]` flattened, interleaved (cos, sin).
pub struct Freqs {
    pub half_dim: usize,
    pub data: Vec<f32>, // seq * half_dim * 2
}

impl Freqs {
    pub fn cos_sin(&self, pos: usize, k: usize) -> (f32, f32) {
        let i = (pos * self.half_dim + k) * 2;
        (self.data[i], self.data[i + 1])
    }
}

/// Port of `precompute_freqs_cis(dim, seqlen, original_seq_len, base,
/// factor, beta_fast, beta_slow)`. YaRN blending active iff
/// `original_seq_len > 0`.
pub fn precompute_freqs(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Freqs {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
        .collect();

    if original_seq_len > 0 {
        // correction range computed in f64 like the Python (math.log)
        let corr_dim = |rot: f64| -> f64 {
            let d = dim as f64;
            let b = base as f64;
            let msl = original_seq_len as f64;
            d * (msl / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * b.ln())
        };
        let lo = corr_dim(beta_fast as f64).floor().max(0.0);
        let mut hi = corr_dim(beta_slow as f64).ceil().min((dim - 1) as f64);
        if lo == hi {
            hi += 0.001;
        }
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = (((i as f32) - lo as f32) / (hi as f32 - lo as f32)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }

    let mut data = Vec::with_capacity(seqlen * half * 2);
    for t in 0..seqlen {
        for &f in &freqs {
            let ang = t as f32 * f;
            data.push(ang.cos());
            data.push(ang.sin());
        }
    }
    Freqs {
        half_dim: half,
        data,
    }
}

/// Rotate the last `rd` dims of each row of `x` (pairs (even, odd) as
/// complex) by the frequencies for absolute position `pos`. `x` rows are
/// laid out [.., row_dim] with the rotary slice being the LAST `rd` elements
/// of each row (V4 applies rope to `x[..., -rd:]`). Output values are
/// bf16-rounded like the reference's in-place `y.copy_(x_f32_rotated)`.
pub fn apply_rope_row(row: &mut [f32], freqs: &Freqs, pos: usize, rd: usize, inverse: bool) {
    let n = row.len();
    let start = n - rd;
    let half = rd / 2;
    debug_assert_eq!(freqs.half_dim, half);
    for k in 0..half {
        let (c, mut s) = freqs.cos_sin(pos, k);
        if inverse {
            s = -s;
        }
        let a = row[start + 2 * k];
        let b = row[start + 2 * k + 1];
        row[start + 2 * k] = to_bf16(a * c - b * s);
        row[start + 2 * k + 1] = to_bf16(a * s + b * c);
    }
}

#[cfg(test)]
mod freqs_dump {
    use super::precompute_freqs;

    /// Dumps `Freqs.data` at GLM-5.2's real rope dims (`qk_rope_head_dim=64`,
    /// `rope_theta=8e6`, `original_seq_len=0` so no YaRN — matches the
    /// `precompute_freqs` call in `glm/loader.rs`) as raw little-endian f32
    /// bytes, so an OpenVINO exporter's cos/sin table can be diffed
    /// bit-for-bit against this crate's table — the one place an OV attention
    /// export must be numerically EXACT, not just ULP-close (a silently
    /// wrong-basis rope table breaks decode while short-prompt parity stays
    /// green).
    ///
    /// Writes to `GLM5_ROPE_FREQS_DUMP` if set, else `$TMPDIR/glm5_rope_freqs_dump.bin`.
    /// Run: `cargo test -p cascadia-engine-sparse-moe --lib dump_real_dims_freqs -- --nocapture`.
    #[test]
    fn dump_real_dims_freqs() {
        const DIM: usize = 64;
        const SEQLEN: usize = 16;
        const THETA: f32 = 8_000_000.0;
        let freqs = precompute_freqs(DIM, SEQLEN, 0, THETA, 1.0, 32.0, 1.0);
        assert_eq!(freqs.data.len(), SEQLEN * (DIM / 2) * 2);

        let path = std::env::var("GLM5_ROPE_FREQS_DUMP").unwrap_or_else(|_| {
            std::env::temp_dir()
                .join("glm5_rope_freqs_dump.bin")
                .to_string_lossy()
                .into_owned()
        });
        let mut bytes = Vec::with_capacity(freqs.data.len() * 4);
        for v in &freqs.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).expect("write rope freqs dump");
        println!(
            "wrote {} f32 values ({} bytes) to {path}",
            freqs.data.len(),
            bytes.len()
        );
        println!("first 8: {:?}", &freqs.data[..8.min(freqs.data.len())]);
    }
}
