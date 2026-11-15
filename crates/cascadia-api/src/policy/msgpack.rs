//! The msgpack subset openpi's `msgpack_numpy` speaks: a zero-copy reader and
//! a writer that picks the same encodings as Python's `msgpack.packb`.

use super::RequestError;

/// Nesting a request needs is 3 (request, images, array); anything deeper is refused.
const MAX_DEPTH: usize = 8;

/// Values a message may hold. A three-camera request with noise is about 80
/// (a numpy array is ~13), so this leaves headroom while keeping the decoded
/// tree small however the message's bytes are spent (a MiB of 1-byte nils
/// would otherwise be a million nodes).
pub(crate) const MAX_VALUES: usize = 512;

/// Entries in one map: the request has 5 keys, `images` one per camera, an
/// array header 4.
pub(crate) const MAX_MAP_LEN: usize = 32;

/// Elements in one array: the only arrays in a request are numpy shapes.
pub(crate) const MAX_ARRAY_LEN: usize = 8;

#[derive(Debug, PartialEq)]
pub(crate) enum Val<'a> {
    Nil,
    Bool(bool),
    UInt(u64),
    Int(i64),
    Float(f64),
    Str(&'a str),
    Bin(&'a [u8]),
    Array(Vec<Val<'a>>),
    Map(Vec<(Val<'a>, Val<'a>)>),
}

impl<'a> Val<'a> {
    /// A map key: msgpack_numpy writes bin keys, Python dicts str keys.
    pub(crate) fn key(&self) -> Option<&'a str> {
        match *self {
            Val::Str(s) => Some(s),
            Val::Bin(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    pub(crate) fn as_u64(&self) -> Option<u64> {
        match *self {
            Val::UInt(u) => Some(u),
            Val::Int(i) => u64::try_from(i).ok(),
            _ => None,
        }
    }
}

/// Parse one msgpack value that spans all of `buf`.
pub(crate) fn parse(buf: &[u8]) -> Result<Val<'_>, RequestError> {
    let mut r = Reader {
        buf,
        pos: 0,
        values: 0,
    };
    let v = r.value(0)?;
    if r.pos != buf.len() {
        return Err(RequestError::new(format!(
            "{} bytes follow the request map",
            buf.len() - r.pos
        )));
    }
    Ok(v)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    values: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], RequestError> {
        if self.buf.len() - self.pos < n {
            return Err(RequestError::new("truncated msgpack message"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn be<const N: usize>(&mut self) -> Result<[u8; N], RequestError> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn len(&mut self, width: usize) -> Result<usize, RequestError> {
        Ok(match width {
            1 => self.be::<1>()?[0] as usize,
            2 => u16::from_be_bytes(self.be()?) as usize,
            _ => u32::from_be_bytes(self.be()?) as usize,
        })
    }

    fn str(&mut self, n: usize) -> Result<Val<'a>, RequestError> {
        std::str::from_utf8(self.take(n)?)
            .map(Val::Str)
            .map_err(|_| RequestError::new("a msgpack string is not UTF-8"))
    }

    fn array(&mut self, n: usize, depth: usize) -> Result<Val<'a>, RequestError> {
        if n > MAX_ARRAY_LEN {
            return Err(RequestError::new(format!(
                "a msgpack array has {n} elements; at most {MAX_ARRAY_LEN} (numpy shapes)"
            )));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.value(depth + 1)?);
        }
        Ok(Val::Array(v))
    }

    fn map(&mut self, n: usize, depth: usize) -> Result<Val<'a>, RequestError> {
        if n > MAX_MAP_LEN {
            return Err(RequestError::new(format!(
                "a msgpack map has {n} entries; at most {MAX_MAP_LEN}"
            )));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let k = self.value(depth + 1)?;
            v.push((k, self.value(depth + 1)?));
        }
        Ok(Val::Map(v))
    }

    fn value(&mut self, depth: usize) -> Result<Val<'a>, RequestError> {
        if depth > MAX_DEPTH {
            return Err(RequestError::new("msgpack nesting is too deep"));
        }
        self.values += 1;
        if self.values > MAX_VALUES {
            return Err(RequestError::new(format!(
                "the message holds more than {MAX_VALUES} msgpack values"
            )));
        }
        let m = self.be::<1>()?[0];
        Ok(match m {
            0x00..=0x7f => Val::UInt(m as u64),
            0x80..=0x8f => return self.map((m & 0x0f) as usize, depth),
            0x90..=0x9f => return self.array((m & 0x0f) as usize, depth),
            0xa0..=0xbf => return self.str((m & 0x1f) as usize),
            0xc0 => Val::Nil,
            0xc2 => Val::Bool(false),
            0xc3 => Val::Bool(true),
            0xc4..=0xc6 => {
                let n = self.len(1 << (m - 0xc4))?;
                Val::Bin(self.take(n)?)
            }
            0xca => Val::Float(f32::from_be_bytes(self.be()?) as f64),
            0xcb => Val::Float(f64::from_be_bytes(self.be()?)),
            0xcc => Val::UInt(self.be::<1>()?[0] as u64),
            0xcd => Val::UInt(u16::from_be_bytes(self.be()?) as u64),
            0xce => Val::UInt(u32::from_be_bytes(self.be()?) as u64),
            0xcf => Val::UInt(u64::from_be_bytes(self.be()?)),
            0xd0 => Val::Int(i8::from_be_bytes(self.be()?) as i64),
            0xd1 => Val::Int(i16::from_be_bytes(self.be()?) as i64),
            0xd2 => Val::Int(i32::from_be_bytes(self.be()?) as i64),
            0xd3 => Val::Int(i64::from_be_bytes(self.be()?)),
            0xd9..=0xdb => {
                let n = self.len(1 << (m - 0xd9))?;
                return self.str(n);
            }
            0xdc | 0xdd => {
                let n = self.len(if m == 0xdc { 2 } else { 4 })?;
                return self.array(n, depth);
            }
            0xde | 0xdf => {
                let n = self.len(if m == 0xde { 2 } else { 4 })?;
                return self.map(n, depth);
            }
            0xe0..=0xff => Val::Int((m as i8) as i64),
            // 0xc1 is never used; 0xc7-0xc9 and 0xd4-0xd8 are extension types.
            _ => {
                return Err(RequestError::new(format!(
                    "unsupported msgpack type 0x{m:02x}"
                )))
            }
        })
    }
}

/// Encodes with the smallest form, as Python's `msgpack.packb` does
/// (`use_bin_type=True`, floats as f64).
#[derive(Default)]
pub(crate) struct Writer {
    pub(crate) buf: Vec<u8>,
}

impl Writer {
    fn head(&mut self, fix: u8, fix_max: usize, codes: [u8; 3], n: usize) {
        if n <= fix_max {
            self.buf.push(fix | n as u8);
        } else if codes[0] != 0 && n <= 0xff {
            self.buf.extend([codes[0], n as u8]);
        } else if n <= 0xffff {
            self.buf.push(codes[1]);
            self.buf.extend((n as u16).to_be_bytes());
        } else {
            self.buf.push(codes[2]);
            self.buf.extend((n as u32).to_be_bytes());
        }
    }

    pub(crate) fn map(&mut self, n: usize) {
        self.head(0x80, 15, [0, 0xde, 0xdf], n);
    }

    pub(crate) fn array(&mut self, n: usize) {
        self.head(0x90, 15, [0, 0xdc, 0xdd], n);
    }

    pub(crate) fn str(&mut self, s: &str) {
        self.head(0xa0, 31, [0xd9, 0xda, 0xdb], s.len());
        self.buf.extend(s.as_bytes());
    }

    pub(crate) fn bin(&mut self, b: &[u8]) {
        if b.len() <= 0xff {
            self.buf.extend([0xc4, b.len() as u8]);
        } else if b.len() <= 0xffff {
            self.buf.push(0xc5);
            self.buf.extend((b.len() as u16).to_be_bytes());
        } else {
            self.buf.push(0xc6);
            self.buf.extend((b.len() as u32).to_be_bytes());
        }
        self.buf.extend(b);
    }

    pub(crate) fn nil(&mut self) {
        self.buf.push(0xc0);
    }

    pub(crate) fn bool(&mut self, b: bool) {
        self.buf.push(if b { 0xc3 } else { 0xc2 });
    }

    pub(crate) fn uint(&mut self, u: u64) {
        if u < 0x80 {
            self.buf.push(u as u8);
        } else if u <= 0xff {
            self.buf.extend([0xcc, u as u8]);
        } else if u <= 0xffff {
            self.buf.push(0xcd);
            self.buf.extend((u as u16).to_be_bytes());
        } else if u <= 0xffff_ffff {
            self.buf.push(0xce);
            self.buf.extend((u as u32).to_be_bytes());
        } else {
            self.buf.push(0xcf);
            self.buf.extend(u.to_be_bytes());
        }
    }

    pub(crate) fn int(&mut self, i: i64) {
        if i >= 0 {
            self.uint(i as u64);
        } else if i >= -32 {
            self.buf.push(i as u8);
        } else if i >= i8::MIN as i64 {
            self.buf.extend([0xd0, i as u8]);
        } else if i >= i16::MIN as i64 {
            self.buf.push(0xd1);
            self.buf.extend((i as i16).to_be_bytes());
        } else if i >= i32::MIN as i64 {
            self.buf.push(0xd2);
            self.buf.extend((i as i32).to_be_bytes());
        } else {
            self.buf.push(0xd3);
            self.buf.extend(i.to_be_bytes());
        }
    }

    pub(crate) fn f64(&mut self, x: f64) {
        self.buf.push(0xcb);
        self.buf.extend(x.to_be_bytes());
    }

    /// A JSON value as msgpack, keys in the map's own order.
    pub(crate) fn json(&mut self, v: &serde_json::Value) {
        use serde_json::Value as J;
        match v {
            J::Null => self.nil(),
            J::Bool(b) => self.bool(*b),
            J::Number(n) => match (n.as_u64(), n.as_i64(), n.as_f64()) {
                (Some(u), _, _) => self.uint(u),
                (_, Some(i), _) => self.int(i),
                (_, _, f) => self.f64(f.unwrap_or(f64::NAN)),
            },
            J::String(s) => self.str(s),
            J::Array(a) => {
                self.array(a.len());
                a.iter().for_each(|x| self.json(x));
            }
            J::Object(o) => {
                self.map(o.len());
                for (k, x) in o {
                    self.str(k);
                    self.json(x);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(f: impl Fn(&mut Writer)) -> Vec<u8> {
        let mut w = Writer::default();
        f(&mut w);
        w.buf
    }

    #[test]
    fn writer_picks_python_encodings() {
        assert_eq!(round(|w| w.uint(5)), [0x05]);
        assert_eq!(round(|w| w.uint(200)), [0xcc, 200]);
        assert_eq!(round(|w| w.uint(300)), [0xcd, 0x01, 0x2c]);
        assert_eq!(round(|w| w.int(-1)), [0xff]);
        assert_eq!(round(|w| w.int(-33)), [0xd0, 0xdf]);
        assert_eq!(round(|w| w.str("ab")), [0xa2, b'a', b'b']);
        assert_eq!(round(|w| w.str(&"x".repeat(32)))[..2], [0xd9, 32]);
        assert_eq!(round(|w| w.bin(&[1, 2])), [0xc4, 2, 1, 2]);
        assert_eq!(round(|w| w.array(16))[..3], [0xdc, 0, 16]);
        assert_eq!(round(|w| w.map(2)), [0x82]);
    }

    #[test]
    fn reader_round_trips_the_writer() {
        let bytes = round(|w| {
            w.map(3);
            w.str("a");
            w.int(-70000);
            w.bin(b"k");
            w.array(2);
            w.f64(1.5);
            w.nil();
            w.str("u");
            w.uint(u64::MAX);
        });
        let Val::Map(m) = parse(&bytes).unwrap() else {
            panic!("not a map")
        };
        assert_eq!(m[0], (Val::Str("a"), Val::Int(-70000)));
        assert_eq!(m[1].0.key(), Some("k"));
        assert_eq!(m[1].1, Val::Array(vec![Val::Float(1.5), Val::Nil]));
        assert_eq!(m[2].1.as_u64(), Some(u64::MAX));
    }

    #[test]
    fn reader_refuses_malformed_input() {
        let e = |b: &[u8]| parse(b).unwrap_err().0;
        assert!(e(&[0xc6, 0xff, 0xff, 0xff, 0xff]).contains("truncated"));
        assert!(e(&[0x93, 0xc0]).contains("truncated"));
        assert!(e(&[0xc7, 0x01, 0x00, 0x00]).contains("unsupported"));
        assert!(e(&[0x01, 0x02]).contains("follow"));
        assert!(e(&[0xa1, 0xff]).contains("UTF-8"));
        assert!(e(&[0x91; 32]).contains("too deep"));
    }

    #[test]
    fn reader_bounds_the_decoded_tree_by_structure() {
        let e = |b: &[u8]| parse(b).unwrap_err().0;
        // 100k one-byte nils fit well under the payload cap.
        let mut nils = vec![0xdd];
        nils.extend(100_000u32.to_be_bytes());
        nils.extend(std::iter::repeat_n(0xc0, 100_000));
        assert!(e(&nils).contains("100000 elements; at most 8"));
        let mut map = vec![0xde];
        map.extend(1000u16.to_be_bytes());
        map.extend(std::iter::repeat_n(0xc0, 2000));
        assert!(e(&map).contains("1000 entries; at most 32"));
        // Every container within its limit, the total over the budget:
        // 30 keys each holding an 8-element array is 30 * 10 values.
        let mut wide = vec![0x80 | 2];
        for _ in 0..2 {
            wide.push(0xc0);
            wide.extend([0xde, 0, 30]);
            for _ in 0..30 {
                wide.push(0xc0);
                wide.push(0x98);
                wide.extend([0xc0; 8]);
            }
        }
        assert!(e(&wide).contains("more than 512 msgpack values"));
    }
}
