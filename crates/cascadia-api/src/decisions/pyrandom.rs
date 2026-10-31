//! CPython's `random.Random` (MT19937), only as far as JEV's permute plan uses it: serve_decide
//! seeds `random.Random(zlib.crc32(question.encode()))` and draws `rng.sample(range(n), n)` three
//! times (`s1_dist`), so the orders, and hence the probabilities, depend on reproducing CPython
//! word for word: `random_seed` / `init_by_array` (Modules/_randommodule.c), `getrandbits`,
//! `_randbelow_with_getrandbits` and `sample`'s pool branch (Lib/random.py).

/// `zlib.crc32(bytes)`: the reflected IEEE CRC-32 (polynomial 0xEDB88320), as an unsigned u32.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    idx: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative int below 2^32. `random_seed` splits abs(seed)
    /// into 32-bit little-endian words, at least one (0 is the key `[0]`), and runs
    /// `init_by_array` over them: one word for every u32.
    pub fn new(seed: u32) -> Self {
        let mut r = Self::init_genrand(19_650_218);
        let key = [seed];
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                r.mt[0] = r.mt[N - 1];
                i = 1;
            }
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, idx: N }
    }

    /// `genrand_uint32`: the next tempered MT19937 word.
    pub fn genrand_uint32(&mut self) -> u32 {
        const MAG01: [u32; 2] = [0, 0x9908_B0DF];
        if self.idx >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7FFF_FFFF);
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for 0 < k <= 32: one word, its top k bits.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        debug_assert!((1..=32).contains(&k));
        self.genrand_uint32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)` for n > 0: redraw `n.bit_length()` bits until below n.
    /// n = 1 still draws (and keeps drawing until it gets a 0), as CPython does.
    pub fn randbelow(&mut self, n: usize) -> usize {
        debug_assert!(n > 0 && n <= u32::MAX as usize);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }

    /// `sample(range(n), n)`. With k == n CPython always takes the pool branch: it does when
    /// `n <= setsize`, where `setsize = 21` plus, for k > 5, `4 ** ceil(log(3k, 4))` >= 3k
    /// (Lib/random.py `sample`, 3.14), so n <= 5 fits 21 and n > 5 fits 3n.
    pub fn sample_all(&mut self, n: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let j = self.randbelow(n - i);
            out.push(pool[j]);
            pool[j] = pool[n - i - 1];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_zlib() {
        // python3 -c 'import zlib; print([zlib.crc32(q.encode()) for q in
        //     ["", "Which team should handle this ticket?", "Wähle die Kategorie ✓"]])'
        assert_eq!(crc32(b""), 0);
        assert_eq!(
            crc32("Which team should handle this ticket?".as_bytes()),
            2887938576
        );
        assert_eq!(crc32("Wähle die Kategorie ✓".as_bytes()), 3633458284);
    }

    #[test]
    fn words_bits_and_randbelow_match_cpython() {
        // python3 -c 'import random
        // for s in (0, 1, 0xdeadbeef):
        //     r = random.Random(s); print(s, [r.getrandbits(32) for _ in range(3)],
        //         [r.getrandbits(5) for _ in range(3)], [r._randbelow(17) for _ in range(3)])'
        for (seed, words, bits5, below17) in [
            (
                0u32,
                [3626764237u32, 1654615998, 3255389356],
                [28u32, 13, 1],
                [8usize, 16, 15],
            ),
            (
                1,
                [577090037, 2444712010, 3639700191],
                [25, 24, 2],
                [8, 3, 15],
            ),
            (
                0xdeadbeef,
                [82178386, 1831771319, 2017055984],
                [28, 19, 29],
                [13, 12, 14],
            ),
        ] {
            let mut r = PyRandom::new(seed);
            assert_eq!([0; 3].map(|_| r.getrandbits(32)), words, "seed {seed}");
            assert_eq!([0; 3].map(|_| r.getrandbits(5)), bits5, "seed {seed}");
            assert_eq!([0; 3].map(|_| r.randbelow(17)), below17, "seed {seed}");
        }
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        question: String,
        crc32: u32,
        n: usize,
        samples: Vec<Vec<usize>>,
    }

    #[test]
    fn three_samples_match_cpython_for_each_fixture_question() {
        // testdata/pyrandom_sample.json was generated with Python 3.14.5:
        // python3 -c 'import random,zlib,json
        // out=[]
        // for q in ["", "Which team should handle this ticket?", "Wähle die Kategorie ✓"]:
        //     for n in (17, 40, 256):
        //         rng = random.Random(zlib.crc32(q.encode()))
        //         out.append({"question": q, "crc32": zlib.crc32(q.encode()), "n": n,
        //                     "samples": [rng.sample(range(n), n) for _ in range(3)]})
        // print(json.dumps(out, ensure_ascii=False))' > testdata/pyrandom_sample.json
        let fx: Vec<Fixture> =
            serde_json::from_str(include_str!("testdata/pyrandom_sample.json")).unwrap();
        assert_eq!(fx.len(), 9);
        for f in fx {
            assert_eq!(crc32(f.question.as_bytes()), f.crc32, "{:?}", f.question);
            let mut r = PyRandom::new(f.crc32);
            for (k, want) in f.samples.iter().enumerate() {
                assert_eq!(
                    &r.sample_all(f.n),
                    want,
                    "{:?} n={} sample {k}",
                    f.question,
                    f.n
                );
            }
        }
    }
}
