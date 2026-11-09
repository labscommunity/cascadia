//! A minimal GGUF metadata reader: one string value by key, such as a
//! model's `tokenizer.chat_template`.

use std::io::{Read, Seek, SeekFrom};

/// The metadata key a GGUF keeps its chat template under.
pub const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";

/// The longest string read into memory (a key or the value asked for).
const MAX_STRING: usize = 1 << 20;

/// How deep arrays of arrays may nest.
const MAX_DEPTH: u32 = 8;

/// The string stored under `key` in a GGUF's (v2 or v3) metadata,
/// `Ok(None)` when it is not there. Reads stop at the key; other values are
/// skipped without being read into memory (a vocabulary's token array
/// included). A corrupt or truncated file is an error, never a panic or an
/// unbounded allocation: no string longer than 1 MiB is read.
pub fn read_string<R: Read + Seek>(r: &mut R, key: &str) -> Result<Option<String>, String> {
    let end = r.seek(SeekFrom::End(0)).map_err(io)?;
    r.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut reader = Reader { r, pos: 0, end };
    let mut magic = [0u8; 4];
    reader.read(&mut magic)?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = reader.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("GGUF version {version} is not supported"));
    }
    let _tensors = reader.u64()?;
    let kvs = reader.u64()?;
    for _ in 0..kvs {
        let k = reader.string()?;
        let ty = reader.u32()?;
        if k == key {
            if ty != STRING {
                return Err(format!("`{key}` is not a string (type {ty})"));
            }
            return reader.string().map(Some);
        }
        reader.skip_value(ty, 0)?;
    }
    Ok(None)
}

/// The GGUF value type of a string, and of an array.
const STRING: u32 = 8;
const ARRAY: u32 = 9;

/// A skip up to this long reads through the buffer rather than seeking (a
/// seek drops a `BufReader`'s buffer, and a vocabulary has ~150k strings).
const SKIP_BY_READ: u64 = 64 << 10;

/// The size of a fixed-size GGUF value type; `None` for strings, arrays and
/// unknown types.
fn fixed_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,
    }
}

struct Reader<'a, R> {
    r: &'a mut R,
    /// Where the next read starts.
    pos: u64,
    /// The stream's length: a skip past it is a truncated file.
    end: u64,
}

impl<R: Read + Seek> Reader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<(), String> {
        self.r.read_exact(buf).map_err(io)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// A length-prefixed string, held to [`MAX_STRING`].
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        if len > MAX_STRING as u64 {
            return Err(format!(
                "a GGUF string of {len} bytes is over the 1 MiB limit"
            ));
        }
        let mut bytes = vec![0u8; len as usize];
        self.read(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| "a GGUF string is not UTF-8".to_owned())
    }

    /// Move `n` bytes on, refusing to pass the end of the stream.
    fn skip(&mut self, n: u64) -> Result<(), String> {
        let to = self
            .pos
            .checked_add(n)
            .filter(|to| *to <= self.end)
            .ok_or("truncated GGUF file")?;
        if n <= SKIP_BY_READ {
            let copied =
                std::io::copy(&mut (&mut *self.r).take(n), &mut std::io::sink()).map_err(io)?;
            if copied != n {
                return Err("truncated GGUF file".into());
            }
        } else {
            self.r.seek(SeekFrom::Start(to)).map_err(io)?;
        }
        self.pos = to;
        Ok(())
    }

    fn skip_value(&mut self, ty: u32, depth: u32) -> Result<(), String> {
        if let Some(n) = fixed_size(ty) {
            return self.skip(n);
        }
        match ty {
            STRING => {
                let len = self.u64()?;
                self.skip(len)
            }
            ARRAY => {
                if depth >= MAX_DEPTH {
                    return Err("GGUF arrays nest too deep".into());
                }
                let elem = self.u32()?;
                let count = self.u64()?;
                if let Some(n) = fixed_size(elem) {
                    let total = n.checked_mul(count).ok_or("truncated GGUF file")?;
                    return self.skip(total);
                }
                if elem != STRING && elem != ARRAY {
                    return Err(format!("unknown GGUF value type {elem}"));
                }
                for _ in 0..count {
                    self.skip_value(elem, depth + 1)?;
                }
                Ok(())
            }
            other => Err(format!("unknown GGUF value type {other}")),
        }
    }
}

fn io(e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        "truncated GGUF file".into()
    } else {
        format!("reading GGUF: {e}")
    }
}

/// The chat template a GGUF file carries, if any.
pub fn chat_template(path: &std::path::Path) -> Result<Option<String>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    read_string(&mut std::io::BufReader::new(file), CHAT_TEMPLATE_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const STRING: u32 = 8;
    const ARRAY: u32 = 9;
    const U32: u32 = 4;
    const BOOL: u32 = 7;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF header (v3) and `kvs`, each a key and its typed value bytes.
    fn gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((kvs.len() as u64).to_le_bytes());
        for (key, ty, value) in kvs {
            string(&mut out, key);
            out.extend(ty.to_le_bytes());
            out.extend(value);
        }
        out
    }

    fn str_value(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        string(&mut v, s);
        v
    }

    /// A string array of `n` tokens, as `tokenizer.ggml.tokens` is.
    fn tokens(n: usize) -> Vec<u8> {
        let mut v = STRING.to_le_bytes().to_vec();
        v.extend((n as u64).to_le_bytes());
        for i in 0..n {
            string(&mut v, &format!("tok{i}"));
        }
        v
    }

    fn read(bytes: Vec<u8>) -> Result<Option<String>, String> {
        read_string(&mut Cursor::new(bytes), CHAT_TEMPLATE_KEY)
    }

    /// The key is found past scalars, a large string array and a nested
    /// array.
    #[test]
    fn the_template_is_found_past_other_metadata() {
        let mut nested = ARRAY.to_le_bytes().to_vec();
        nested.extend(2u64.to_le_bytes());
        for _ in 0..2 {
            nested.extend(U32.to_le_bytes());
            nested.extend(3u64.to_le_bytes());
            nested.extend([0u8; 12]);
        }
        let bytes = gguf(&[
            ("general.architecture", STRING, str_value("qwen3")),
            ("qwen3.block_count", U32, 28u32.to_le_bytes().to_vec()),
            ("tokenizer.ggml.add_bos_token", BOOL, vec![1]),
            ("tokenizer.ggml.tokens", ARRAY, tokens(50_000)),
            ("nested", ARRAY, nested),
            (CHAT_TEMPLATE_KEY, STRING, str_value("{{ messages }}")),
            ("after", STRING, str_value("x")),
        ]);
        assert_eq!(read(bytes), Ok(Some("{{ messages }}".into())));
    }

    #[test]
    fn a_missing_key_is_none() {
        let bytes = gguf(&[("general.architecture", STRING, str_value("qwen3"))]);
        assert_eq!(read(bytes), Ok(None));
    }

    /// A file cut short, a bad magic, an unknown version or value type, and
    /// a string too long to be a template are errors, not panics.
    #[test]
    fn a_corrupt_file_is_an_error() {
        let whole = gguf(&[
            ("tokenizer.ggml.tokens", ARRAY, tokens(10)),
            (CHAT_TEMPLATE_KEY, STRING, str_value("{{ messages }}")),
        ]);
        for cut in [0, 3, 10, 30, whole.len() - 3] {
            assert!(read(whole[..cut].to_vec()).is_err(), "cut at {cut}");
        }
        let mut bad = whole.clone();
        bad[0] = b'X';
        assert!(read(bad).is_err());
        let mut v1 = whole.clone();
        v1[4] = 1;
        assert!(read(v1).is_err());
        assert!(read(gguf(&[("odd", 99, vec![])])).is_err());
        let mut huge = Vec::new();
        huge.extend((u64::MAX).to_le_bytes());
        assert!(read(gguf(&[(CHAT_TEMPLATE_KEY, STRING, huge)])).is_err());
        let mut long = Vec::new();
        long.extend(((MAX_STRING + 1) as u64).to_le_bytes());
        long.extend(vec![b'x'; MAX_STRING + 1]);
        assert!(read(gguf(&[(CHAT_TEMPLATE_KEY, STRING, long)])).is_err());
        // A huge array count over fixed-size elements runs off the end.
        let mut many = U32.to_le_bytes().to_vec();
        many.extend((u64::MAX / 2).to_le_bytes());
        assert!(read(gguf(&[("many", ARRAY, many)])).is_err());
    }
}
