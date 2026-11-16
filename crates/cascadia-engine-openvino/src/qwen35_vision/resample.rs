//! Bicubic (and bilinear) resize that reproduces Pillow's
//! `Image.resize(..., BICUBIC | BILINEAR)` on 8-bit RGB bit for bit (Pillow `libImaging/Resample.c`, unchanged in
//! behaviour since 7.0; the fixtures were made with 12.3).
//!
//! Hugging Face's Qwen2-VL/Qwen3.5 PIL processor resizes through Pillow, and
//! the model was trained on that, so matching it exactly removes one source
//! of drift. The algorithm, as Pillow implements it:
//!
//! - separable: a horizontal pass, then a vertical pass, each rounding back
//!   to `u8` (the intermediate image is 8-bit, not float);
//! - a Keys cubic kernel with `a = -0.5`, support 2 (bilinear: the triangle
//!   `1 - |x|`, support 1), widened by the scale factor when downsampling
//!   (antialiasing);
//! - per-output-pixel weights normalised to sum 1 in `f64`, then converted to
//!   22-bit fixed point, accumulated in integers with a rounding bias.

/// Pillow's `PRECISION_BITS` for 8-bit images: 32 - 8 - 2.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Pillow's resampling filters this port reproduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// `Resampling.BILINEAR` (2): Gemma 3's SigLIP processor.
    Bilinear,
    /// `Resampling.BICUBIC` (3): Qwen, Gemma 4, Pixtral.
    Bicubic,
}

impl Filter {
    /// The filter for a `PILImageResampling` code, if this port has it.
    pub fn from_pil(code: u64) -> Option<Self> {
        match code {
            2 => Some(Self::Bilinear),
            3 => Some(Self::Bicubic),
            _ => None,
        }
    }

    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Self::Bicubic => bicubic(x),
        }
    }
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per output index: first input index and fixed-point weights.
struct Coeffs {
    ksize: usize,
    /// `(xmin, count)` per output index.
    bounds: Vec<(usize, usize)>,
    /// `out_size * ksize` fixed-point weights.
    kk: Vec<i32>,
}

/// Pillow's `precompute_coeffs` + `normalize_coeffs_8bpc` for the full input
/// range (`box` = whole image).
fn precompute(in_size: usize, out_size: usize, filter: Filter) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let ss = 1.0 / filterscale;
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // C casts truncate toward zero; clamping after matches Pillow.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut ww = 0.0;
        for (x, kx) in k.iter_mut().enumerate().take(xmax) {
            let w = filter.weight((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        for (x, kx) in k.iter().enumerate().take(xmax) {
            let v = if ww != 0.0 { kx / ww } else { *kx };
            let fixed = v * f64::from(1u32 << PRECISION_BITS);
            kk[xx * ksize + x] = if v < 0.0 {
                (-0.5 + fixed) as i32
            } else {
                (0.5 + fixed) as i32
            };
        }
        bounds.push((xmin, xmax));
    }
    Coeffs { ksize, bounds, kk }
}

#[inline]
fn clip8(ss: i64) -> u8 {
    (ss >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resize packed RGB8 `src` (`w x h`) to `nw x nh`, exactly as Pillow does.
///
/// Pillow >= 12 (`Image.resize`) shrinks an image more than 100 times taller
/// than wide in two calls, height first and then width; the result differs
/// from the single two-pass resize by up to an 8-bit level, so it is
/// mirrored here. HF's processors resize through whatever Pillow is
/// installed; the fixtures were made with 12.3.
pub fn resize_rgb8_bicubic(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    resize_rgb8(src, w, h, nw, nh, Filter::Bicubic)
}

/// [`resize_rgb8_bicubic`] with any of Pillow's [`Filter`]s.
pub fn resize_rgb8(
    src: &[u8],
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
    filter: Filter,
) -> Vec<u8> {
    assert_eq!(src.len(), w * h * 3, "src is not {w}x{h} RGB8");
    assert!(nw > 0 && nh > 0, "empty target size");
    if h > w * 100 && nh < h {
        let tall = resize_two_pass(src, w, h, w, nh, filter);
        return resize_two_pass(&tall, w, nh, nw, nh, filter);
    }
    resize_two_pass(src, w, h, nw, nh, filter)
}

/// Pillow's `ImagingResampleInner`: a horizontal pass, then a vertical one.
fn resize_two_pass(
    src: &[u8],
    w: usize,
    h: usize,
    nw: usize,
    nh: usize,
    filter: Filter,
) -> Vec<u8> {
    let need_h = nw != w;
    let need_v = nh != h;
    let bias: i64 = 1 << (PRECISION_BITS - 1);

    // Horizontal pass over only the rows the vertical pass will read (Pillow's
    // `ybox_first..ybox_last`); the result is identical to a full pass.
    let vert = need_v.then(|| precompute(h, nh, filter));
    let (row0, row1) = match &vert {
        Some(c) => {
            let first = c.bounds[0].0;
            let (lmin, lcount) = c.bounds[nh - 1];
            (first, lmin + lcount)
        }
        None => (0, h),
    };
    let tmp_h = row1 - row0;
    let tmp: Vec<u8> = if need_h {
        let c = precompute(w, nw, filter);
        let mut out = vec![0u8; nw * tmp_h * 3];
        for y in 0..tmp_h {
            let src_row = &src[(y + row0) * w * 3..(y + row0 + 1) * w * 3];
            let dst_row = &mut out[y * nw * 3..(y + 1) * nw * 3];
            for xx in 0..nw {
                let (xmin, count) = c.bounds[xx];
                let k = &c.kk[xx * c.ksize..xx * c.ksize + count];
                for ch in 0..3 {
                    let mut ss = bias;
                    for (x, &kx) in k.iter().enumerate() {
                        ss += i64::from(src_row[(x + xmin) * 3 + ch]) * i64::from(kx);
                    }
                    dst_row[xx * 3 + ch] = clip8(ss);
                }
            }
        }
        out
    } else {
        src[row0 * w * 3..row1 * w * 3].to_vec()
    };
    let tw = if need_h { nw } else { w };

    let Some(c) = vert else {
        return tmp;
    };
    let mut out = vec![0u8; nw * nh * 3];
    for yy in 0..nh {
        let (ymin, count) = c.bounds[yy];
        let ymin = ymin - row0;
        let k = &c.kk[yy * c.ksize..yy * c.ksize + count];
        for x in 0..tw {
            for ch in 0..3 {
                let mut ss = bias;
                for (y, &ky) in k.iter().enumerate() {
                    ss += i64::from(tmp[((y + ymin) * tw + x) * 3 + ch]) * i64::from(ky);
                }
                out[(yy * tw + x) * 3 + ch] = clip8(ss);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_is_a_copy() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|v| v as u8).collect();
        assert_eq!(resize_rgb8_bicubic(&src, 4, 3, 4, 3), src);
    }

    #[test]
    fn flat_colour_stays_flat() {
        let src = [10u8, 200, 77].repeat(37 * 23);
        for (nw, nh) in [(64, 64), (5, 3), (37, 50), (90, 23)] {
            let out = resize_rgb8_bicubic(&src, 37, 23, nw, nh);
            assert!(out.chunks(3).all(|p| p == [10, 200, 77]), "{nw}x{nh}");
        }
    }

    #[test]
    fn weights_sum_to_one_in_fixed_point() {
        for (i, o) in [(1000, 320), (40, 320), (777, 768), (5, 2)] {
            for f in [Filter::Bicubic, Filter::Bilinear] {
                let c = precompute(i, o, f);
                for xx in 0..o {
                    let (_, n) = c.bounds[xx];
                    let s: i64 = c.kk[xx * c.ksize..xx * c.ksize + n]
                        .iter()
                        .map(|&v| i64::from(v))
                        .sum();
                    assert!(
                        (s - (1 << PRECISION_BITS)).abs() <= n as i64,
                        "{i}->{o} at {xx}: {s}"
                    );
                }
            }
        }
    }
}
