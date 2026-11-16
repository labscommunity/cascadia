//! The policy codec against bytes packed by the real `openpi_client`
//! (`tools/vla_reference.py msgpack`), and every decoding refusal.

use cascadia_api::policy::{
    check_reply, decode_request, encode_metadata, encode_reply, encode_request, max_payload_bytes,
    metadata_json,
};
use cascadia_types::{ActionChunk, NoiseMode, PolicySpec, ResizeConvention, RtcInputs};

const REQUEST: &[u8] = include_bytes!("fixtures/policy/request.msgpack");
const REQUEST_F8: &[u8] = include_bytes!("fixtures/policy/request-f8-state.msgpack");
const REPLY: &[u8] = include_bytes!("fixtures/policy/reply.msgpack");
const REQUEST_RTC: &[u8] = include_bytes!("fixtures/policy/request-rtc.msgpack");

fn spec() -> PolicySpec {
    PolicySpec {
        model: "mock".into(),
        device: "CPU".into(),
        cameras: vec!["image".into(), "wrist".into()],
        height: 4,
        width: 4,
        state_len: 3,
        chunk: [2, 3],
        noise: NoiseMode::Input,
        noise_shape: Some(vec![2, 3]),
        resize: ResizeConvention::TopLeft,
        prompt_max_bytes: 64,
        attribution: None,
        manifest_sha256: String::new(),
        precision: String::new(),
        server_resize: None,
        rtc_chunk_size: None,
    }
}

/// [`spec`] for an RTC package: leftover `[2, 3]`, so a horizon of at most 1.
fn rtc_spec() -> PolicySpec {
    PolicySpec {
        rtc_chunk_size: Some(2),
        ..spec()
    }
}

fn frame(invert: bool) -> Vec<u8> {
    (0..48u8)
        .map(|b| if invert { 255 - b } else { b })
        .collect()
}

/// A tiny msgpack writer for hand-built requests, mirroring msgpack_numpy.
mod mp {
    pub fn map(b: &mut Vec<u8>, n: u8) {
        b.push(0x80 | n);
    }
    pub fn str(b: &mut Vec<u8>, s: &str) {
        assert!(s.len() < 32);
        b.push(0xa0 | s.len() as u8);
        b.extend(s.as_bytes());
    }
    pub fn bin(b: &mut Vec<u8>, d: &[u8]) {
        b.push(0xc6);
        b.extend((d.len() as u32).to_be_bytes());
        b.extend(d);
    }
    pub fn uint(b: &mut Vec<u8>, u: u64) {
        b.push(0xcf);
        b.extend(u.to_be_bytes());
    }
    pub fn ndarray(b: &mut Vec<u8>, dtype: &str, shape: &[u64], data: &[u8]) {
        map(b, 4);
        bin(b, b"__ndarray__");
        b.push(0xc3);
        bin(b, b"data");
        bin(b, data);
        bin(b, b"dtype");
        str(b, dtype);
        bin(b, b"shape");
        b.push(0x90 | shape.len() as u8);
        for &d in shape {
            uint(b, d);
        }
    }
}

/// `(dtype, shape, data)` of one array.
type Array = (&'static str, Vec<u64>, Vec<u8>);

/// Request parts, each overridable, packed in openpi's layout.
struct Req {
    images: Vec<(&'static str, &'static str, Vec<u64>, Vec<u8>)>,
    state: (&'static str, Vec<u64>, Vec<u8>),
    prompt: Option<String>,
    noise: Option<(&'static str, Vec<u64>, Vec<u8>)>,
    extra: Option<&'static str>,
    /// Leftover, inference_delay (`None`: left out), max_guidance_weight, execution_horizon.
    rtc: Option<(Array, Option<u64>, f64, u64)>,
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Req {
    fn good() -> Self {
        Req {
            images: vec![
                ("image", "|u1", vec![4, 4, 3], frame(false)),
                ("wrist", "|u1", vec![4, 4, 3], frame(true)),
            ],
            state: ("<f4", vec![3], f32s(&[0.5, -1.0, 2.25])),
            prompt: Some("pick up the cube".into()),
            noise: Some(("<f4", vec![1, 2, 3], f32s(&[0.0; 6]))),
            extra: None,
            rtc: None,
        }
    }

    fn pack(&self) -> Vec<u8> {
        let mut b = Vec::new();
        let n = 2
            + self.prompt.is_some() as u8
            + self.noise.is_some() as u8
            + self.extra.is_some() as u8
            + self.rtc.as_ref().map_or(0, |r| 3 + r.1.is_some() as u8);
        mp::map(&mut b, n);
        mp::str(&mut b, "images");
        mp::map(&mut b, self.images.len() as u8);
        for (name, dtype, shape, data) in &self.images {
            mp::str(&mut b, name);
            mp::ndarray(&mut b, dtype, shape, data);
        }
        mp::str(&mut b, "state");
        mp::ndarray(&mut b, self.state.0, &self.state.1, &self.state.2);
        if let Some(p) = &self.prompt {
            mp::str(&mut b, "prompt");
            mp::str(&mut b, p);
        }
        if let Some((dtype, shape, data)) = &self.noise {
            mp::str(&mut b, "noise");
            mp::ndarray(&mut b, dtype, shape, data);
        }
        if let Some(k) = self.extra {
            mp::str(&mut b, k);
            b.push(0xc0);
        }
        if let Some(((dtype, shape, data), delay, weight, horizon)) = &self.rtc {
            mp::str(&mut b, "prev_chunk_left_over");
            mp::ndarray(&mut b, dtype, shape, data);
            if let Some(d) = delay {
                mp::str(&mut b, "inference_delay");
                mp::uint(&mut b, *d);
            }
            mp::str(&mut b, "max_guidance_weight");
            b.push(0xcb);
            b.extend(weight.to_be_bytes());
            mp::str(&mut b, "execution_horizon");
            mp::uint(&mut b, *horizon);
        }
        b
    }

    fn err(&self, spec: &PolicySpec) -> String {
        decode_request(&self.pack(), spec).unwrap_err().0
    }
}

#[test]
fn decodes_openpi_client_request_bytes() {
    let obs = decode_request(REQUEST, &spec()).unwrap();
    assert_eq!(obs.images.len(), 2);
    assert_eq!(obs.images[0].0, "image");
    assert_eq!(obs.images[0].1.data, frame(false));
    assert_eq!(obs.images[1].0, "wrist");
    assert_eq!(obs.images[1].1.data, frame(true));
    assert_eq!(obs.state, vec![0.5, -1.0, 2.25]);
    assert_eq!(obs.prompt, "pick up the cube");
    assert_eq!(obs.seed, Some(7));
    assert_eq!(obs.noise, Some(vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0]));
}

#[test]
fn decodes_f8_state_and_numpy_scalar_seed() {
    let obs = decode_request(REQUEST_F8, &spec()).unwrap();
    assert_eq!(obs.state, vec![0.5, -1.0, 2.25]);
    assert_eq!(obs.seed, Some(7));
    assert_eq!(obs.noise, None);
}

#[test]
fn decodes_openpi_client_rtc_fields() {
    let obs = decode_request(REQUEST_RTC, &rtc_spec()).unwrap();
    assert_eq!(
        obs.rtc,
        Some(RtcInputs {
            prev_chunk_left_over: vec![0.25, -0.5, 1.0, 0.0, 0.0, 0.0],
            inference_delay: 1,
            max_guidance_weight: 5.0,
            execution_horizon: 1,
        })
    );
    assert_eq!(obs.noise, None);
    // A package exported without RTC refuses them rather than ignoring them.
    let e = decode_request(REQUEST_RTC, &spec()).unwrap_err().0;
    assert!(e.contains("takes no RTC inputs"), "{e}");
    // Without the fields an RTC package serves the plain chunk.
    assert_eq!(decode_request(REQUEST, &rtc_spec()).unwrap().rtc, None);
}

#[test]
fn rtc_fields_round_trip_and_take_a_batch_of_one() {
    let mut obs = decode_request(REQUEST_RTC, &rtc_spec()).unwrap();
    obs.noise = Some(vec![0.5; 6]);
    let back = decode_request(&encode_request(&obs, &rtc_spec()).unwrap(), &rtc_spec()).unwrap();
    assert_eq!(back, obs);
    // physicalai's runtime sends the leftover as [1, T, D].
    let rtc = |left: Array, delay: Option<u64>| {
        let mut r = Req::good();
        r.rtc = Some((left, delay, 2.5, 1));
        r
    };
    let left = || ("<f4", vec![1, 2, 3], f32s(&[1.0; 6]));
    let o = decode_request(&rtc(left(), Some(0)).pack(), &rtc_spec()).unwrap();
    assert_eq!(o.rtc.unwrap().prev_chunk_left_over, vec![1.0; 6]);
    let e = rtc(("<f4", vec![3, 3], f32s(&[1.0; 9])), Some(0)).err(&rtc_spec());
    assert!(
        e.contains("prev_chunk_left_over has shape [3, 3]; expected [2, 3]"),
        "{e}"
    );
    let e = rtc(("<f8", vec![2, 3], vec![0; 48]), Some(0)).err(&rtc_spec());
    assert!(
        e.contains("prev_chunk_left_over has dtype <f8; expected <f4"),
        "{e}"
    );
    // All four or none.
    let e = rtc(left(), None).err(&rtc_spec());
    assert!(e.contains("RTC needs all of"), "{e}");
    // The ranges are the engine's.
    let mut r = rtc(left(), Some(2));
    r.rtc.as_mut().unwrap().3 = 1;
    assert!(r
        .err(&rtc_spec())
        .contains("inference_delay <= execution_horizon"));
}

#[test]
fn encodes_the_reply_openpi_client_packs() {
    let chunk = ActionChunk {
        actions: vec![0.5, -1.25, 3.0, 0.001, 2.0, -0.0],
        steps: 2,
        dim: 3,
        infer_ms: 12.5,
    };
    assert_eq!(encode_reply(&chunk), REPLY);
}

#[test]
fn hand_built_request_decodes() {
    decode_request(&Req::good().pack(), &spec()).unwrap();
}

#[test]
fn refuses_bad_images() {
    let s = spec();
    let mut r = Req::good();
    r.images[0].1 = "<f4";
    assert!(r.err(&s).contains("dtype <f4; expected |u1"));
    let mut r = Req::good();
    r.images[0].2 = vec![4, 3, 4];
    assert!(r.err(&s).contains("shape [4, 3, 4]; expected [4, 4, 3]"));
    let mut r = Req::good();
    r.images[0].3.pop();
    assert!(r
        .err(&s)
        .contains("47 data bytes; expected 48 for dtype |u1 and shape [4, 4, 3]"));
    let mut r = Req::good();
    r.images.pop();
    assert!(r.err(&s).contains("missing camera \"wrist\""));
    let mut r = Req::good();
    r.images[1].0 = "top";
    assert!(r.err(&s).contains("unknown camera \"top\""));
}

#[test]
fn refuses_bad_state() {
    let s = spec();
    let mut r = Req::good();
    r.state = ("<i4", vec![3], vec![0; 12]);
    assert!(r.err(&s).contains("dtype <i4; expected <f4 or <f8"));
    let mut r = Req::good();
    r.state = ("<f4", vec![4], f32s(&[0.0; 4]));
    assert!(r.err(&s).contains("shape [4]; expected [3]"));
    let mut r = Req::good();
    r.state = ("<f8", vec![3], f32s(&[0.0; 3]));
    assert!(r.err(&s).contains("12 data bytes; expected 24"));
    let mut r = Req::good();
    r.state = ("<f4", vec![3], f32s(&[0.0, f32::INFINITY, 0.0]));
    assert!(r.err(&s).contains("non-finite"));
}

#[test]
fn refuses_bad_noise() {
    let s = spec();
    let mut r = Req::good();
    r.noise = Some(("<f8", vec![1, 2, 3], vec![0; 48]));
    assert!(r.err(&s).contains("noise has dtype <f8; expected <f4"));
    let mut r = Req::good();
    r.noise = Some(("<f4", vec![2, 3], f32s(&[0.0; 6])));
    assert!(r.err(&s).contains("shape [2, 3]; expected [1, 2, 3]"));
    let mut r = Req::good();
    r.noise = Some(("<f4", vec![1, 2, 3], f32s(&[0.0; 5])));
    assert!(r.err(&s).contains("20 data bytes; expected 24"));
    let mut r = Req::good();
    r.noise = Some(("<f4", vec![1, 2, 3], f32s(&[0.0, 0.0, 10.5, 0.0, 0.0, 0.0])));
    assert!(r.err(&s).contains("|x| <= 10"));
    let mut r = Req::good();
    r.noise = Some((
        "<f4",
        vec![1, 2, 3],
        f32s(&[0.0, 0.0, f32::NAN, 0.0, 0.0, 0.0]),
    ));
    assert!(r.err(&s).contains("finite"));
    let mut constant = spec();
    constant.noise = NoiseMode::Constant;
    constant.noise_shape = None;
    assert!(Req::good()
        .err(&constant)
        .contains("takes no noise input (noise mode constant)"));
}

#[test]
fn refuses_overflowing_shapes_before_allocating() {
    // Each dim fits, the product does not: refused by checked arithmetic,
    // not by an allocation.
    let mut big = spec();
    big.noise_shape = Some(vec![u64::MAX as usize, u64::MAX as usize]);
    let mut r = Req::good();
    r.noise = Some(("<f4", vec![1, u64::MAX, u64::MAX], vec![]));
    assert!(r.err(&big).contains("an overflowing count"));
}

#[test]
fn refuses_bad_prompt_and_keys() {
    let s = spec();
    let mut r = Req::good();
    r.prompt = Some("x".repeat(31));
    let mut short = spec();
    short.prompt_max_bytes = 30;
    assert!(r
        .err(&short)
        .contains("prompt is 31 bytes; the limit is 30"));
    let mut r = Req::good();
    r.prompt = None;
    assert!(r.err(&s).contains("missing key \"prompt\""));
    let mut r = Req::good();
    r.extra = Some("observation/state");
    assert!(r.err(&s).contains("unknown key \"observation/state\""));
    let mut r = Req::good();
    r.extra = Some("seed");
    // A nil seed is accepted as absent.
    decode_request(&r.pack(), &s).unwrap();
    let mut dup = Req::good().pack();
    dup[0] += 1;
    dup.extend([0xa6]);
    dup.extend(b"prompt");
    dup.extend([0xa1, b'x']);
    assert!(decode_request(&dup, &s)
        .unwrap_err()
        .0
        .contains("sent twice"));
    assert!(decode_request(&[0x93, 1, 2, 3], &s)
        .unwrap_err()
        .0
        .contains("one msgpack map"));
    assert!(decode_request(&REQUEST[..REQUEST.len() - 1], &s)
        .unwrap_err()
        .0
        .contains("truncated"));
}

#[test]
fn refuses_non_finite_replies() {
    let mut c = ActionChunk {
        actions: vec![0.0; 6],
        steps: 2,
        dim: 3,
        infer_ms: 1.0,
    };
    check_reply(&c).unwrap();
    c.actions[4] = f32::INFINITY;
    assert!(check_reply(&c).unwrap_err().0.contains("non-finite"));
    c.actions.pop();
    assert!(check_reply(&c).unwrap_err().0.contains("5 actions"));
}

#[test]
fn metadata_has_every_spec_field() {
    let s = spec();
    let json: serde_json::Value = serde_json::from_str(&metadata_json(&s)).unwrap();
    assert_eq!(json["cameras"], serde_json::json!(["image", "wrist"]));
    assert_eq!(json["noise"], "input");
    assert_eq!(json["noise_shape"], serde_json::json!([2, 3]));
    assert_eq!(json["resize"], "top_left");
    let packed = encode_metadata(&s);
    // A map16 of the 15 spec fields (`attribution` omitted when absent, and
    // `rtc_chunk_size` among them) plus policy_name and chunk_size; keys are
    // msgpack str, as Python dicts pack.
    assert_eq!(packed[..3], [0xde, 0, 17]);
    assert_eq!(json["manifest_sha256"], "");
    assert_eq!(json["precision"], "");
    assert!(json["server_resize"].is_null());
    assert_eq!(json["chunk_size"], s.chunk[0]);
    assert!(packed.windows(8).any(|w| w == b"top_left"));
}

#[test]
fn socket_metadata_is_the_same_on_every_load() {
    use cascadia_api::policy::{decode_metadata, encode_metadata_for, PackageIdentity};
    let mut s = spec();
    s.manifest_sha256 = "cd".repeat(32);
    s.precision = "f16".into();
    let load = |ms| PackageIdentity {
        policy_name: "smolvla".into(),
        warmup_ms: ms,
    };
    let first = encode_metadata_for(&s, &load(Some(180)));
    assert_eq!(first, encode_metadata_for(&s, &load(Some(2310))));
    assert_eq!(first, encode_metadata_for(&s, &load(None)));
    assert!(!first.windows(9).any(|w| w == b"warmup_ms"));
    // The package and the precision do change it.
    let mut f32 = s.clone();
    f32.precision = "f32".into();
    assert_ne!(first, encode_metadata_for(&f32, &load(Some(180))));
    let mut other = s.clone();
    other.manifest_sha256 = "ef".repeat(32);
    assert_ne!(first, encode_metadata_for(&other, &load(Some(180))));
    let (back, id) = decode_metadata(&first).unwrap();
    assert_eq!(back, s);
    assert_eq!(id.policy_name, "smolvla");
    assert_eq!(id.warmup_ms, None);
}

/// The non-RTC metadata is byte for byte what 2522a4d57 (before RTC) sent, so
/// a package's metadata hash is unchanged. The fixtures were captured from
/// that commit in both serde_json map modes; the test picks the one matching
/// the mode this build compiled (run it with `--features
/// serde_json/preserve_order` for the insertion-ordered one).
#[test]
fn non_rtc_metadata_bytes_are_unchanged_in_both_map_modes() {
    use cascadia_api::policy::{encode_metadata_for, metadata_json_for, PackageIdentity};
    let s = PolicySpec {
        model: "smolvla-libero".into(),
        device: "GPU".into(),
        cameras: vec!["image".into(), "image2".into()],
        height: 512,
        width: 512,
        state_len: 8,
        chunk: [50, 7],
        noise: NoiseMode::Input,
        noise_shape: Some(vec![50, 32]),
        resize: ResizeConvention::TopLeft,
        prompt_max_bytes: 1024,
        attribution: None,
        manifest_sha256: "ab".repeat(32),
        precision: "f16".into(),
        server_resize: Some(cascadia_types::ServerResize::Letterbox),
        rtc_chunk_size: None,
    };
    let id = PackageIdentity {
        policy_name: "smolvla".into(),
        warmup_ms: Some(180),
    };
    let mut probe = serde_json::Map::new();
    probe.insert("b".into(), 1.into());
    probe.insert("a".into(), 2.into());
    let ordered = probe.keys().next().map(String::as_str) == Some("b");
    let (bytes, json): (&[u8], &str) = if ordered {
        (
            include_bytes!("fixtures/policy/metadata-nortc-ordered.msgpack"),
            include_str!("fixtures/policy/metadata-nortc-ordered.json"),
        )
    } else {
        (
            include_bytes!("fixtures/policy/metadata-nortc-sorted.msgpack"),
            include_str!("fixtures/policy/metadata-nortc-sorted.json"),
        )
    };
    eprintln!(
        "serde_json map mode: {}",
        if ordered { "preserve_order" } else { "sorted" }
    );
    assert_eq!(encode_metadata_for(&s, &id), bytes);
    assert_eq!(metadata_json_for(&s, &id), json);
}

#[test]
fn rtc_fields_on_a_package_without_rtc_are_not_encoded() {
    let mut obs = decode_request(REQUEST_RTC, &rtc_spec()).unwrap();
    obs.noise = None;
    let e = encode_request(&obs, &spec()).unwrap_err();
    assert!(e.0.contains("takes no RTC inputs"), "{e}");
}

#[test]
fn max_payload_bytes_follows_the_formula() {
    let s = spec();
    // 2 * 4*4*3 + 3 * 8 + 6 * 4 + 64 + 64 KiB.
    assert_eq!(max_payload_bytes(&s), 96 + 24 + 24 + 64 + 65536);
    // An RTC package's leftover, [2, 3] f32.
    assert_eq!(max_payload_bytes(&rtc_spec()), max_payload_bytes(&s) + 24);
    let json: serde_json::Value = serde_json::from_str(&metadata_json(&rtc_spec())).unwrap();
    assert_eq!(json["rtc_chunk_size"], 2);
    let mut big = spec();
    big.height = 2048;
    big.width = 2048;
    assert_eq!(max_payload_bytes(&big), cascadia_engine::PAYLOAD_MAX_BYTES);
}

/// X-VLA: the model emits 20-wide ee6d actions and noise; a postprocessor
/// serves 7. Metadata and replies carry the served width; the message cap
/// counts the 20-wide noise a request sends.
#[test]
fn served_action_width_is_independent_of_the_noise_width() {
    let s = PolicySpec {
        cameras: vec!["image".into(), "image2".into()],
        height: 224,
        width: 224,
        state_len: 8,
        chunk: [30, 7],
        noise_shape: Some(vec![30, 20]),
        ..spec()
    };
    let json: serde_json::Value = serde_json::from_str(&metadata_json(&s)).unwrap();
    assert_eq!(json["chunk"], serde_json::json!([30, 7]));
    assert_eq!(json["noise_shape"], serde_json::json!([30, 20]));
    assert_eq!(
        max_payload_bytes(&s),
        2 * 224 * 224 * 3 + 8 * 8 + 30 * 20 * 4 + 64 + 65536
    );
    let mut c = ActionChunk {
        actions: vec![0.5; 210],
        steps: 30,
        dim: 7,
        infer_ms: 1.0,
    };
    check_reply(&c).unwrap();
    // The ndarray header's shape: fixarray of 2, then 30 and 7.
    assert!(encode_reply(&c).windows(3).any(|w| w == [0x92, 30, 7]));
    c.actions = vec![0.5; 600];
    assert!(check_reply(&c)
        .unwrap_err()
        .0
        .contains("600 actions for a [30, 7] chunk"));
}
