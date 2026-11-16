//! The physicalai codec against bytes physicalai's own `_protocol.py` packed
//! (`tests/fixtures/physicalai/generate.py`), and every refusal.

use cascadia_api::policy::physicalai::{
    area_frame, area_frame_f32, cubic_frame, decode_control, decode_predict, encode_error,
    encode_metadata, encode_predict_reply, encode_reset_ok, read_seq, resize_frame,
    resize_frame_f32, smart_resize, ControlError, ErrorCode, ServerIdentity, Timings,
    MAX_REQUEST_BYTES, NO_SEQ, XR0_CUBIC_EXACT,
};
use cascadia_api::policy::{max_payload_bytes, metadata_json_for, PackageIdentity};
use cascadia_types::{
    ActionChunk, Image, NoiseMode, PolicySpec, ResizeConvention, RtcInputs, ServerResize,
};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/physicalai/", $name)).as_slice()
    };
}

const SERVER_ID: &str = "0123456789abcdef0123456789abcdef";

fn spec() -> PolicySpec {
    PolicySpec {
        model: "mock".into(),
        device: "CPU".into(),
        cameras: vec!["image".into(), "wrist".into()],
        height: 16,
        width: 16,
        state_len: 3,
        chunk: [2, 3],
        noise: NoiseMode::Input,
        noise_shape: Some(vec![2, 3]),
        resize: ResizeConvention::TopLeft,
        prompt_max_bytes: 64,
        attribution: None,
        manifest_sha256: "ab".repeat(32),
        precision: "f32".into(),
        server_resize: Some(cascadia_types::ServerResize::Letterbox),
        rtc_chunk_size: None,
    }
}

/// [`spec`] exported with RTC: leftover `[2, 3]`.
fn rtc_spec() -> PolicySpec {
    PolicySpec {
        rtc_chunk_size: Some(2),
        ..spec()
    }
}

fn max() -> u64 {
    u64::from(max_payload_bytes(&spec()))
}

fn frames() -> (Vec<u8>, Vec<u8>) {
    let f = fixture!("frames.rgb");
    (f[..768].to_vec(), f[768..].to_vec())
}

fn identity() -> ServerIdentity {
    ServerIdentity {
        server_id: SERVER_ID.into(),
        name: "cell1".into(),
        policy_name: "smolvla".into(),
        manifest_sha256: "ab".repeat(32),
        backend: "openvino".into(),
        device: "CPU".into(),
    }
}

#[test]
fn metadata_reply_is_byte_for_byte_physicalai() {
    assert_eq!(max(), 67184, "generate.py's MAX_REQUEST_BYTES");
    assert_eq!(
        encode_metadata(&identity(), &spec(), max()),
        fixture!("metadata.msgpack")
    );
}

#[test]
fn rtc_metadata_is_byte_for_byte_physicalai() {
    let max = u64::from(max_payload_bytes(&rtc_spec()));
    assert_eq!(max, 67184 + 24, "generate.py's MAX_REQUEST_BYTES_RTC");
    assert_eq!(
        encode_metadata(&identity(), &rtc_spec(), max),
        fixture!("metadata-rtc.msgpack")
    );
}

/// What physicalai's `RTCExecution` sends: the leftover `[1, T, D]` and numpy scalars.
#[test]
fn rtc_execution_inputs_decode() {
    let max = u64::from(max_payload_bytes(&rtc_spec()));
    let r = decode_predict(fixture!("predict-rtc.msgpack"), &rtc_spec(), max).unwrap();
    assert_eq!(r.seq, 7);
    assert_eq!(
        r.observation.rtc,
        Some(RtcInputs {
            prev_chunk_left_over: vec![0.75, 0.5, -1.0, 0.0, 0.0, 0.0],
            inference_delay: 0,
            max_guidance_weight: 5.0,
            execution_horizon: 1,
        })
    );
    let e = decode_predict(fixture!("predict-rtc.msgpack"), &spec(), max).unwrap_err();
    assert!(e.message().contains("takes no RTC inputs"), "{e}");
}

#[test]
fn rtc_refusals() {
    let max = u64::from(max_payload_bytes(&rtc_spec()));
    let scalar = |dtype: &str, data: &[u8]| {
        let (dtype, data) = (dtype.to_string(), data.to_vec());
        val(move |b| mp::np(b, &dtype, &[], &data))
    };
    let req = || {
        good()
            .input(
                "prev_chunk_left_over",
                val(|b| mp::np(b, "float32", &[1, 2, 3], &f32s(&[0.5; 6]))),
            )
            .input("inference_delay", scalar("int64", &1i64.to_le_bytes()))
            .input(
                "max_guidance_weight",
                scalar("float32", &5f32.to_le_bytes()),
            )
            .input("execution_horizon", scalar("int64", &1i64.to_le_bytes()))
    };
    let err = |r: Req| decode_predict(&r.pack(), &rtc_spec(), max).unwrap_err().0;
    decode_predict(&req().pack(), &rtc_spec(), max).unwrap();
    let e = err(req().without_input("execution_horizon"));
    assert!(e.contains("RTC needs all of"), "{e}");
    let e = err(req().input("inference_delay", scalar("float32", &1f32.to_le_bytes())));
    assert!(
        e.contains("inference_delay must be a non-negative integer"),
        "{e}"
    );
    let e = err(req().input("execution_horizon", scalar("int64", &(-1i64).to_le_bytes())));
    assert!(
        e.contains("execution_horizon must be a non-negative integer"),
        "{e}"
    );
    let e = err(req().input(
        "prev_chunk_left_over",
        val(|b| mp::np(b, "float32", &[1, 2, 8], &f32s(&[0.5; 16]))),
    ));
    assert!(
        e.contains("prev_chunk_left_over has shape [1, 2, 8]"),
        "{e}"
    );
    let e = err(req().input("execution_horizon", scalar("int64", &2i64.to_le_bytes())));
    assert!(e.contains("more than half the chunk"), "{e}");
    // Peer-chosen dtypes and sizes are refused, never read past the scalar.
    for (dtype, bytes) in [
        ("complex128", 16),
        ("complex64", 8),
        ("float16", 2),
        ("bool", 1),
    ] {
        let e = err(req().input("max_guidance_weight", scalar(dtype, &vec![1; bytes])));
        assert!(
            e.contains("max_guidance_weight must be a single number"),
            "{dtype}: {e}"
        );
        let e = err(req().input("inference_delay", scalar(dtype, &vec![1; bytes])));
        assert!(e.contains("inference_delay must be"), "{dtype}: {e}");
    }
    let two = val(|b| mp::np(b, "int64", &[2], &[0; 16]));
    let e = err(req().input("execution_horizon", two));
    assert!(
        e.contains("execution_horizon must be a single number"),
        "{e}"
    );
    // Big-endian data is refused, never read as little-endian (`>f4` 5.0 would be 5.7e-41).
    for (dtype, bytes) in [
        (">f4", 5f32.to_be_bytes().to_vec()),
        (">f8", 5f64.to_be_bytes().to_vec()),
        (">i8", 1i64.to_be_bytes().to_vec()),
    ] {
        let e = err(req().input("max_guidance_weight", scalar(dtype, &bytes)));
        assert!(e.contains("big-endian"), "{dtype}: {e}");
    }
    let e = err(req().input(
        "prev_chunk_left_over",
        val(|b| mp::np(b, ">f4", &[1, 2, 3], &[0; 24])),
    ));
    assert!(e.contains("big-endian"), "{e}");
    // Native and explicit little-endian orders are read.
    for dtype in ["<f4", "=f4", "float32"] {
        decode_predict(
            &req()
                .input("max_guidance_weight", scalar(dtype, &5f32.to_le_bytes()))
                .pack(),
            &rtc_spec(),
            max,
        )
        .unwrap_or_else(|e| panic!("{dtype}: {e}"));
    }
    // A package without RTC refuses the scalars before reading them.
    let e = decode_predict(
        &good()
            .input("max_guidance_weight", scalar("complex128", &[1; 16]))
            .pack(),
        &spec(),
        max,
    )
    .unwrap_err()
    .0;
    assert!(e.contains("takes no RTC inputs"), "{e}");
}

#[test]
fn control_requests() {
    decode_control(fixture!("metadata-request.msgpack"), max()).unwrap();
    // {"protocol_version": 2}
    assert_eq!(
        decode_control(
            &[
                0x81, 0xb0, b'p', b'r', b'o', b't', b'o', b'c', b'o', b'l', b'_', b'v', b'e', b'r',
                b's', b'i', b'o', b'n', 2
            ],
            max()
        ),
        Err(ControlError::UnsupportedVersion)
    );
    let mut extra = fixture!("metadata-request.msgpack").to_vec();
    extra[0] = 0x82;
    extra.extend([0xa1, b'x', 0x01]);
    assert!(matches!(
        decode_control(&extra, max()),
        Err(ControlError::Invalid(e)) if e.message().contains("only protocol_version")
    ));
    assert!(matches!(
        decode_control(&[0xc1], max()),
        Err(ControlError::Invalid(_))
    ));
    assert!(matches!(
        decode_control(&[0x01], max()),
        Err(ControlError::Invalid(_))
    ));
}

#[test]
fn raw_and_jpeg_predicts_decode() {
    let (image, wrist) = frames();
    let raw = decode_predict(fixture!("predict-raw.msgpack"), &spec(), max()).unwrap();
    assert_eq!((raw.seq, raw.budget_ms), (3, 2000));
    let o = &raw.observation;
    assert_eq!(o.state, [0.5, -1.0, 2.25]);
    assert_eq!(o.prompt, "pick up the cube");
    assert_eq!(o.noise, None);
    assert_eq!(o.images[0].0, "image");
    assert_eq!(o.images[0].1.data, image);
    assert_eq!(o.images[1].0, "wrist");
    assert_eq!(o.images[1].1.data, wrist);

    let jpeg = decode_predict(fixture!("predict-jpeg.msgpack"), &spec(), max()).unwrap();
    // OpenCV (libjpeg-turbo) decoded the same bytes in physicalai's server.
    let cv = fixture!("predict-jpeg.decoded.rgb");
    let ours: Vec<u8> = jpeg
        .observation
        .images
        .iter()
        .flat_map(|(_, i)| i.data.clone())
        .collect();
    let worst = ours
        .iter()
        .zip(cv)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    eprintln!("JPEG decode vs OpenCV: worst channel difference {worst}");
    assert!(worst <= 2, "our JPEG decode differs from OpenCV by {worst}");
}

#[test]
fn unbatched_frames_f64_state_and_noise() {
    let r = decode_predict(fixture!("predict-unbatched-noise.msgpack"), &spec(), max()).unwrap();
    assert_eq!(r.seq, 4);
    assert_eq!(r.observation.state, [0.5, -1.0, 2.25]);
    assert_eq!(
        r.observation.noise.unwrap(),
        [0.0, 1.0, -1.0, 0.5, -0.5, 2.0]
    );
    let mut constant = spec();
    constant.noise = NoiseMode::Constant;
    constant.noise_shape = None;
    let e = decode_predict(
        fixture!("predict-unbatched-noise.msgpack"),
        &constant,
        max(),
    )
    .unwrap_err();
    assert!(e.message().contains("no noise input"), "{e}");
}

#[test]
fn a_bare_images_key_is_the_only_camera() {
    let mut one = spec();
    one.cameras = vec!["front".into()];
    let r = decode_predict(fixture!("predict-single-camera.msgpack"), &one, max()).unwrap();
    assert_eq!(r.observation.images[0].0, "front");
    let e = decode_predict(fixture!("predict-single-camera.msgpack"), &spec(), max()).unwrap_err();
    assert!(e.message().contains("one-camera package"), "{e}");
}

#[test]
fn reply_and_errors_are_byte_for_byte_physicalai() {
    let chunk = ActionChunk {
        actions: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        steps: 2,
        dim: 3,
        infer_ms: 250.0,
    };
    let t = Timings {
        queue_s: 0.001,
        compute_s: 0.25,
        decode_s: 0.0005,
    };
    assert_eq!(
        encode_predict_reply(&chunk, 3, SERVER_ID, t),
        fixture!("reply.msgpack")
    );
    assert_eq!(
        encode_error(ErrorCode::InvalidRequest, 3),
        fixture!("error-invalid-request.msgpack")
    );
    assert_eq!(
        encode_error(ErrorCode::UnsupportedVersion, NO_SEQ),
        fixture!("error-unsupported-version.msgpack")
    );
    assert_eq!(encode_reset_ok(), fixture!("reset-ok.msgpack"));
    let codes = [
        (ErrorCode::InvalidRequest, "invalid_request"),
        (ErrorCode::UnsupportedVersion, "unsupported_version"),
        (ErrorCode::ShuttingDown, "shutting_down"),
        (ErrorCode::Superseded, "superseded"),
        (ErrorCode::Expired, "expired"),
        (ErrorCode::ModelError, "model_error"),
    ];
    for (c, s) in codes {
        assert_eq!(c.as_str(), s);
    }
}

#[test]
fn read_seq_takes_the_top_level_integer() {
    assert_eq!(read_seq(fixture!("predict-raw.msgpack")), 3);
    assert_eq!(read_seq(fixture!("metadata-request.msgpack")), NO_SEQ);
    assert_eq!(read_seq(b"\xc1"), NO_SEQ);
    // {"seq": -5}
    assert_eq!(read_seq(&[0x81, 0xa3, b's', b'e', b'q', 0xfb]), -5);
}

#[test]
fn the_websocket_metadata_carries_the_package_identity() {
    let id = PackageIdentity {
        policy_name: "smolvla".into(),
        warmup_ms: Some(180),
    };
    let v: serde_json::Value = serde_json::from_str(&metadata_json_for(&spec(), &id)).unwrap();
    assert_eq!(v["manifest_sha256"], "ab".repeat(32));
    assert_eq!(v["precision"], "f32");
    assert_eq!(v["policy_name"], "smolvla");
    assert_eq!(v["chunk_size"], 2);
    assert!(v["rtc_chunk_size"].is_null());
    assert_eq!(v["warmup_ms"], 180);
    let v: serde_json::Value = serde_json::from_str(&metadata_json_for(&rtc_spec(), &id)).unwrap();
    assert_eq!(v["rtc_chunk_size"], 2);
}

// ---- refusals: hand-built requests in physicalai's layout ----

mod mp {
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
    pub fn int(b: &mut Vec<u8>, i: i64) {
        b.push(0xd3);
        b.extend(i.to_be_bytes());
    }
    pub fn shape(b: &mut Vec<u8>, shape: &[u64]) {
        b.push(0x90 | shape.len() as u8);
        shape.iter().for_each(|&d| uint(b, d));
    }
    pub fn np(b: &mut Vec<u8>, dtype: &str, shape_: &[u64], data: &[u8]) {
        b.push(0x84);
        str(b, "__np__");
        b.push(0xc3);
        str(b, "dtype");
        str(b, dtype);
        str(b, "shape");
        shape(b, shape_);
        str(b, "data");
        bin(b, data);
    }
    pub fn image(b: &mut Vec<u8>, codec: &str, dtype: &str, shape_: &[u64], data: &[u8]) {
        b.push(0x84);
        str(b, "codec");
        str(b, codec);
        str(b, "shape");
        shape(b, shape_);
        str(b, "dtype");
        str(b, dtype);
        str(b, "data");
        bin(b, data);
    }
}

/// Request parts, each overridable; `raw` bytes replace a part verbatim.
#[derive(Clone)]
struct Req {
    version: Vec<u8>,
    seq: Vec<u8>,
    budget: Vec<u8>,
    inputs: Vec<(String, Vec<u8>)>,
    images: Vec<(String, Vec<u8>)>,
    extra: bool,
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn val(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let mut b = Vec::new();
    f(&mut b);
    b
}

fn good() -> Req {
    let (image, wrist) = frames();
    Req {
        version: vec![1],
        seq: vec![3],
        budget: val(|b| mp::uint(b, 2000)),
        inputs: vec![
            (
                "state".into(),
                val(|b| mp::np(b, "float32", &[1, 3], &f32s(&[0.5, -1.0, 2.25]))),
            ),
            (
                "task".into(),
                val(|b| {
                    b.push(0x91);
                    mp::str(b, "pick up the cube");
                }),
            ),
        ],
        images: vec![
            (
                "images.image".into(),
                val(|b| mp::image(b, "raw", "uint8", &[1, 16, 16, 3], &image)),
            ),
            (
                "images.wrist".into(),
                val(|b| mp::image(b, "raw", "uint8", &[16, 16, 3], &wrist)),
            ),
        ],
        extra: false,
    }
}

impl Req {
    fn pack(&self) -> Vec<u8> {
        let mut b = vec![0x80 | (5 + u8::from(self.extra))];
        mp::str(&mut b, "protocol_version");
        b.extend(&self.version);
        mp::str(&mut b, "seq");
        b.extend(&self.seq);
        mp::str(&mut b, "budget_ms");
        b.extend(&self.budget);
        for (key, items) in [("inputs", &self.inputs), ("images", &self.images)] {
            mp::str(&mut b, key);
            b.push(0x80 | items.len() as u8);
            for (k, v) in items {
                mp::str(&mut b, k);
                b.extend(v);
            }
        }
        if self.extra {
            mp::str(&mut b, "x");
            b.push(0xc0);
        }
        b
    }

    fn input(mut self, k: &str, v: Vec<u8>) -> Self {
        self.inputs.retain(|(n, _)| n != k);
        self.inputs.push((k.into(), v));
        self
    }

    fn image(mut self, k: &str, v: Vec<u8>) -> Self {
        self.images.retain(|(n, _)| n != k);
        self.images.push((k.into(), v));
        self
    }

    fn without_input(mut self, k: &str) -> Self {
        self.inputs.retain(|(n, _)| n != k);
        self
    }

    fn refused(&self, want: &str) {
        self.refused_within(max(), want);
    }

    fn refused_within(&self, max_bytes: u64, want: &str) {
        let e = decode_predict(&self.pack(), &spec(), max_bytes).unwrap_err();
        assert!(
            e.message().contains(want),
            "want {want:?}, got {:?}",
            e.message()
        );
    }
}

#[test]
fn the_hand_built_request_decodes() {
    let r = decode_predict(&good().pack(), &spec(), max()).unwrap();
    assert_eq!(r.observation.images.len(), 2);
    assert_eq!(r.observation.state, [0.5, -1.0, 2.25]);
}

#[test]
fn envelope_refusals() {
    let mut r = good();
    r.extra = true;
    r.refused("unexpected or missing fields");
    let mut r = good();
    r.version = vec![2];
    r.refused("Unsupported predict protocol version");
    let mut r = good();
    r.version = vec![0xc3];
    r.refused("Unsupported predict protocol version");
    let mut r = good();
    r.seq = val(|b| mp::int(b, -1));
    r.refused("Invalid prediction sequence");
    let mut r = good();
    r.seq = vec![0xc2];
    r.refused("Invalid prediction sequence");
    let mut r = good();
    r.budget = vec![0];
    r.refused("Invalid prediction budget");
    let mut r = good();
    r.budget = val(|b| mp::uint(b, 60_001));
    r.refused("exceeds the server maximum");
    let e = decode_predict(&[0xc1], &spec(), max()).unwrap_err();
    assert!(
        e.message().contains("Invalid MessagePack protocol map"),
        "{e}"
    );
    let e = decode_predict(&[0x93, 1, 2, 3], &spec(), max()).unwrap_err();
    assert!(e.message().contains("string-keyed"), "{e}");
    let big = good().pack();
    let e = decode_predict(&big, &spec(), big.len() as u64 - 1).unwrap_err();
    assert!(e.message().contains("exceeds"), "{e}");
}

#[test]
fn array_refusals() {
    let np = |dtype: &str, shape: &[u64], data: &[u8]| val(|b| mp::np(b, dtype, shape, data));
    good()
        .input("state", np("float32", &[1, 3], &[0; 11]))
        .refused("do not match its data length");
    good()
        .input("state", np("<U4", &[1], &[0; 4]))
        .refused("Invalid __np__ array dtype");
    good()
        .input("state", np("float32", &[1, 4], &[0; 16]))
        .refused("expected [3] or [1, 3]");
    good()
        .input("state", np("int32", &[3], &[0; 12]))
        .refused("expected float32 or float64");
    good()
        .input("state", np("float32", &[3], &f32s(&[0.0, f32::NAN, 1.0])))
        .refused("non-finite");
    good()
        .input("state", vec![0x01])
        .refused("numeric arrays or list[str]");
    let malformed = val(|b| {
        b.push(0x82);
        mp::str(b, "__np__");
        b.push(0xc3);
        mp::str(b, "dtype");
        mp::str(b, "float32");
    });
    good()
        .input("state", malformed)
        .refused("Malformed __np__ array payload");
    good()
        .without_input("state")
        .refused("missing input \"state\"");
    good()
        .without_input("task")
        .refused("missing input \"task\"");
    good()
        .input("task", val(|b| b.push(0x90)))
        .refused("exactly one string");
    good()
        .input("task", val(|b| mp::str(b, "x")))
        .refused("numeric arrays or list[str]");
    good()
        .input("prompt", val(|b| b.push(0x90)))
        .refused("unknown input \"prompt\"");
    good()
        .input("images.x", val(|b| b.push(0x90)))
        .refused("separate images map");
    let long = val(|b| {
        b.push(0x91);
        b.push(0xd9);
        b.push(65);
        b.extend([b'a'; 65]);
    });
    good().input("task", long).refused("the limit is 64");
}

#[test]
fn image_refusals() {
    let (image, _) = frames();
    let img = |codec: &str, dtype: &str, shape: &[u64], data: &[u8]| {
        val(|b| mp::image(b, codec, dtype, shape, data))
    };
    let s = [16, 16, 3];
    good()
        .image("wrist", img("raw", "uint8", &s, &image))
        .refused("'images' namespace");
    good()
        .image("images.wrist", img("png", "uint8", &s, &image))
        .refused("Malformed image codec");
    good()
        .image("images.wrist", img("raw", "float32", &s, &image))
        .refused("Malformed image codec");
    good()
        .image("images.wrist", vec![0x01])
        .refused("Image entry must be a map");
    good()
        .image("images.wrist", img("raw", "uint8", &[16, 0, 3], &[]))
        .refused("positive integers");
    good()
        .image("images.wrist", img("raw", "uint8", &[256, 3], &[0; 768]))
        .refused("HxWx3");
    good()
        .image(
            "images.wrist",
            img("raw", "uint8", &[16, 16, 4], &[0; 1024]),
        )
        .refused("HxWx3");
    good()
        .image(
            "images.wrist",
            img("raw", "uint8", &[2, 16, 16, 3], &[0; 1536]),
        )
        .refused("batch dimension must be one");
    good()
        .image("images.wrist", img("raw", "uint8", &s, &image[1..]))
        .refused("does not match its shape");
    good()
        .image("images.wrist", img("raw", "uint8", &[4000, 4000, 3], &[]))
        .refused("exceed max_request_bytes");
    good()
        .image("images.wrist", img("jpeg", "uint8", &s, b"nope"))
        .refused("Malformed JPEG image data");
    good()
        .image(
            "images.wrist",
            img("jpeg", "uint8", &s, &[0xff, 0xd8, 0x00, 0x00, 0x00, 0x00]),
        )
        .refused("Malformed JPEG marker");
    good()
        .image(
            "images.wrist",
            img("jpeg", "uint8", &s, &[0xff, 0xd8, 0xff, 0xe0, 0x00, 0x40]),
        )
        .refused("Malformed JPEG segment");
    good()
        .image(
            "images.wrist",
            img("jpeg", "uint8", &s, &[0xff, 0xd8, 0xff, 0xd9, 0xff, 0xd9]),
        )
        .refused("JPEG image has no dimensions");
    // A real 16x16 JPEG declared 8x8, and a truncated one.
    let jpeg = jpeg_frame();
    good()
        .image("images.wrist", img("jpeg", "uint8", &[8, 8, 3], &jpeg))
        .refused("JPEG dimensions do not match");
    good()
        .image(
            "images.wrist",
            img("jpeg", "uint8", &s, &jpeg[..jpeg.len() / 2]),
        )
        .refused("JPEG");
    good()
        .image("images.other", img("raw", "uint8", &s, &image))
        .refused("unknown camera \"other\"");
    good()
        .image("images.wrist", img("raw", "uint8", &[4097, 1, 3], &[]))
        .refused_within(MAX_REQUEST_BYTES, "at most 4096 pixels a side");
    let mut r = good();
    r.images.pop();
    r.refused("missing camera \"wrist\"");
    good()
        .input("images.image", vec![0x01])
        .refused("separate images map");
}

/// The `images.image` JPEG from the physicalai fixture.
fn jpeg_frame() -> Vec<u8> {
    let b = fixture!("predict-jpeg.msgpack");
    let start = b.windows(2).position(|w| w == [0xff, 0xd8]).unwrap();
    let end = start
        + b[start..]
            .windows(2)
            .position(|w| w == [0xff, 0xd9])
            .unwrap()
        + 2;
    b[start..end].to_vec()
}

#[test]
fn a_duplicate_key_is_refused() {
    let mut b = good().pack();
    b[0] += 1;
    mp::str(&mut b, "seq");
    b.push(3);
    let e = decode_predict(&b, &spec(), max()).unwrap_err();
    assert!(
        e.message().contains("unexpected or missing fields") || e.message().contains("twice"),
        "{e}"
    );
}

// ---- frames at another resolution are resized as physicalai's server does ----

fn f32s_of(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn resizes_match_physicalai_for_every_convention() {
    let cases: [(&str, u32, u32, &[u8]); 3] = [
        ("down", 48, 64, fixture!("resize-down.rgb")),
        ("tall", 40, 18, fixture!("resize-tall.rgb")),
        ("up", 6, 10, fixture!("resize-up.rgb")),
    ];
    let convs = [
        (ResizeConvention::TopLeft, "top_left"),
        (ResizeConvention::Center, "center"),
        (ResizeConvention::Stretch, "stretch"),
    ];
    let golden = |name: &str, conv: &str| -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/physicalai/resize-{name}-{conv}.f32",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    };
    for (name, h, w, rgb) in cases {
        let img = Image {
            height: h,
            width: w,
            data: rgb.to_vec(),
        };
        for (conv, label) in convs {
            let want = f32s_of(&golden(name, label));
            let ours = resize_frame_f32(&img, 16, 16, conv);
            let d = worst(&ours, &want);
            eprintln!("resize {name} {h}x{w} {label}: worst |ours - physicalai| = {d} (of 255)");
            assert!(d < 1e-3, "{name} {label}: {d}");
            // The u8 frame the engine gets is within rounding of physicalai's float frame.
            let u8s: Vec<f32> = resize_frame(&img, 16, 16, conv)
                .data
                .iter()
                .map(|&b| f32::from(b))
                .collect();
            assert!(worst(&u8s, &want) <= 0.5 + 1e-3, "{name} {label} u8");
        }
    }
}

#[test]
fn a_native_480x640_jpeg_passes_through() {
    let native = fixture!("predict-native-480x640.msgpack");
    let r = decode_predict(native, &spec(), MAX_REQUEST_BYTES).unwrap();
    assert_eq!(r.seq, 6);
    let frames: Vec<f32> = r
        .observation
        .images
        .iter()
        .inspect(|(_, i)| assert_eq!((i.height, i.width), (16, 16)))
        .flat_map(|(_, i)| i.data.iter().map(|&b| f32::from(b)))
        .collect();
    // physicalai's server: OpenCV's JPEG decode, then the same letterbox. Our JPEG
    // decoder differs from OpenCV's by a level or two before averaging.
    let d = worst(
        &frames,
        &f32s_of(fixture!("predict-native-480x640.top_left.f32")),
    );
    eprintln!("native 480x640 JPEG to 16x16 top_left: worst |ours - physicalai| = {d} (of 255)");
    assert!(d <= 3.0, "{d}");
    // The spec's own payload cap would have refused it.
    assert!(decode_predict(native, &spec(), max()).is_err());
}

#[test]
fn packages_without_a_ported_resize_refuse_other_sizes() {
    let mut s = spec();
    s.server_resize = None;
    let e = decode_predict(
        fixture!("predict-native-480x640.msgpack"),
        &s,
        MAX_REQUEST_BYTES,
    )
    .unwrap_err();
    assert!(e.message().contains("is 480x640"), "{e}");
    assert!(e.message().contains("send frames at 16x16"), "{e}");
    // 480x640 is not square, so max_image_side cannot make it 16x16.
    assert!(!e.message().contains("max_image_side"), "{e}");
    let (image, _) = frames();
    let mut r = good();
    r.images.pop();
    let big: Vec<u8> = image.iter().copied().cycle().take(32 * 32 * 3).collect();
    r = r.image(
        "images.wrist",
        val(|b| mp::image(b, "raw", "uint8", &[32, 32, 3], &big)),
    );
    let e = decode_predict(&r.pack(), &s, MAX_REQUEST_BYTES).unwrap_err();
    assert!(e.message().contains("or set max_image_side=16"), "{e}");
    // Frames at the model size need no resize.
    decode_predict(fixture!("predict-raw.msgpack"), &s, max()).unwrap();
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/physicalai/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn native_frames() -> [(&'static str, u32, u32); 3] {
    [("down", 48, 64), ("tall", 40, 18), ("up", 6, 10)]
}

/// LeRobot's pi0 `resize_with_pad` (torch bilinear, `align_corners=False`,
/// centred pad) is the same letterbox as physicalai's `pi05`.
#[test]
fn pi0_frames_match_lerobots_resize_with_pad() {
    for (name, h, w) in native_frames() {
        let img = Image {
            height: h,
            width: w,
            data: read_fixture(&format!("resize-{name}.rgb")),
        };
        let want = f32s_of(&read_fixture(&format!("resize-{name}-pi0.f32")));
        let d = worst(
            &resize_frame_f32(&img, 16, 16, ResizeConvention::Center),
            &want,
        );
        eprintln!("pi0 {name} {h}x{w}: worst |ours - LeRobot| = {d} (of 255)");
        assert!(d < 1e-3, "{name}: {d}");
    }
}

#[test]
fn vla_jepa_area_frames_match_physicalai() {
    for (name, h, w) in native_frames() {
        let img = Image {
            height: h,
            width: w,
            data: read_fixture(&format!("resize-{name}.rgb")),
        };
        let want = f32s_of(&read_fixture(&format!("resize-{name}-area.f32")));
        let d = worst(&area_frame_f32(&img, 16, 16), &want);
        eprintln!("area {name} {h}x{w}: worst |ours - physicalai| = {d} (of 255)");
        assert!(d < 1e-3, "{name}: {d}");
        let u8s: Vec<f32> = area_frame(&img, 16, 16)
            .data
            .iter()
            .map(|&b| f32::from(b))
            .collect();
        assert!(worst(&u8s, &want) <= 0.5 + 1e-3, "{name} u8");
    }
}

/// physicalai's `xr0._resize_image` (factor 8, 300 pixels in the fixtures):
/// the target size, and OpenCV's `INTER_CUBIC` pixels byte for byte.
#[test]
fn xr0_frames_match_physicalai_byte_for_byte() {
    for (h, w, out) in [
        (20, 20, (16, 16)),
        (40, 40, (16, 16)),
        (12, 12, (16, 16)),
        (44, 52, (8, 16)),
        (30, 20, (16, 8)),
    ] {
        assert_eq!(smart_resize(h, w, 8, 300), Some(out), "{h}x{w}");
        let img = Image {
            height: h,
            width: w,
            data: read_fixture(&format!("xr0-{h}x{w}.rgb")),
        };
        let want = read_fixture(&format!("xr0-{h}x{w}.out-{}x{}.rgb", out.0, out.1));
        let ours = cubic_frame(&img, out.0, out.1).data;
        let differ = ours.iter().zip(&want).filter(|(a, b)| a != b).count();
        eprintln!(
            "xr0 {h}x{w} -> {}x{}: {differ} of {} bytes differ from OpenCV",
            out.0,
            out.1,
            want.len()
        );
        assert_eq!(differ, 0, "{h}x{w}");
    }
    assert_eq!(smart_resize(70, 90, 8, 3000), Some((48, 56)));
    let img = Image {
        height: 70,
        width: 90,
        data: read_fixture("xr0-70x90.rgb"),
    };
    assert_eq!(
        cubic_frame(&img, 48, 56).data,
        read_fixture("xr0-70x90.out-48x56.rgb")
    );
    assert_eq!(smart_resize(250, 260, 32, 90000), Some((256, 256)));
    let img = Image {
        height: 250,
        width: 260,
        data: read_fixture("xr0-250x260.rgb"),
    };
    let ours = cubic_frame(&img, 256, 256).data;
    let want = read_fixture("xr0-250x260.out-256x256.rgb");
    let differ = ours.iter().zip(&want).filter(|(a, b)| a != b).count();
    eprintln!(
        "xr0 250x260 -> 256x256: {differ} of {} bytes differ from OpenCV",
        want.len()
    );
    assert_eq!(differ, 0);
    // LIBERO's 256x256 is left alone; a 480x640 camera becomes 256x320.
    assert_eq!(smart_resize(256, 256, 32, 90000), Some((256, 256)));
    assert_eq!(smart_resize(480, 640, 32, 90000), Some((256, 320)));
    assert_eq!(smart_resize(1, 300, 32, 90000), None);
}

#[test]
fn xr0_takes_frames_its_resize_sends_to_the_export_size() {
    let mut s = spec();
    s.server_resize = Some(ServerResize::SmartResizeCubic {
        factor: 8,
        max_pixels: 300,
    });
    let raw = |h: u64, w: u64, data: Vec<u8>| {
        let mut r = good();
        r.images.clear();
        r = r.image(
            "images.image",
            val(|b| mp::image(b, "raw", "uint8", &[h, w, 3], &data)),
        );
        r.image(
            "images.wrist",
            val(|b| mp::image(b, "raw", "uint8", &[h, w, 3], &data)),
        )
    };
    if !XR0_CUBIC_EXACT {
        // x86: physicalai's resize is Intel IPP's, which is not reproduced.
        let e = decode_predict(
            &raw(20, 20, read_fixture("xr0-20x20.rgb")).pack(),
            &s,
            MAX_REQUEST_BYTES,
        )
        .unwrap_err();
        assert!(e.message().contains("Intel IPP"), "{e}");
        assert!(e.message().contains("send frames at 16x16"), "{e}");
        return;
    }
    let ok = decode_predict(
        &raw(20, 20, read_fixture("xr0-20x20.rgb")).pack(),
        &s,
        MAX_REQUEST_BYTES,
    )
    .unwrap();
    assert_eq!(
        ok.observation.images[0].1.data,
        read_fixture("xr0-20x20.out-16x16.rgb")
    );
    let e = decode_predict(
        &raw(30, 20, read_fixture("xr0-30x20.rgb")).pack(),
        &s,
        MAX_REQUEST_BYTES,
    )
    .unwrap_err();
    assert!(e.message().contains("turns into 16x8"), "{e}");
    assert!(e.message().contains("send frames at 16x16"), "{e}");
}
