use cascadia_diffusion::{
    manifest::{Dtype, Manifest, Port},
    wire,
};
#[test]
fn shared_python_manifest_digest_and_binary_frame() {
    let manifest: Manifest = serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    assert_eq!(
        manifest.id().unwrap(),
        include_str!("fixtures/manifest.sha256").trim()
    );
    let bytes = include_bytes!("fixtures/frame.bin");
    let (header, values) = wire::decode(bytes).unwrap();
    assert_eq!(header.manifest, manifest.id().unwrap());
    assert_eq!(values, vec![&[0u8, 0, 128, 63, 0, 0, 0, 192][..]]);
    // Encoding may reorder JSON keys; decoding must preserve the contract.
    let encoded = wire::encode(&header, &values).unwrap();
    assert_eq!(wire::decode(&encoded).unwrap(), (header, values));
    for n in 0..bytes.len() {
        assert!(wire::decode(&bytes[..n]).is_err());
    }
    let mut extra = bytes.to_vec();
    extra.push(0);
    assert!(wire::decode(&extra).is_err());
    let mut nan = bytes.to_vec();
    let n = nan.len();
    nan[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(wire::decode(&nan).is_err());
    let mut bad_length = bytes.to_vec();
    bad_length[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(wire::decode(&bad_length).is_err());
}
#[test]
fn tensor_size_and_schema_cannot_overflow_or_alias() {
    for shape in [vec![usize::MAX, 2], vec![0], vec![1024; 9], vec![1 << 28]] {
        assert!(Port {
            name: "input".into(),
            dtype: Dtype::F32,
            shape
        }
        .bytes()
        .is_err());
    }
    let (mut header, _) = wire::decode(include_bytes!("fixtures/frame.bin")).unwrap();
    header.tensors.push(header.tensors[0].clone());
    assert!(header.validate().is_err());
}
#[test]
fn artifact_tampering_missing_files_and_traversal_fail_closed() {
    let mut manifest: Manifest =
        serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("unet")).unwrap();
    std::fs::write(root.path().join("unet/model.xml"), b"xml").unwrap();
    std::fs::write(root.path().join("unet/model.bin"), b"bin").unwrap();
    let digest = manifest.id().unwrap();
    let selected = ["unet".into()];
    manifest
        .verify_bundle(root.path(), &digest, &selected)
        .unwrap();
    std::fs::write(root.path().join("unet/model.bin"), b"bad").unwrap();
    assert!(manifest
        .verify_bundle(root.path(), &digest, &selected)
        .is_err());
    std::fs::remove_file(root.path().join("unet/model.bin")).unwrap();
    assert!(manifest
        .verify_bundle(root.path(), &digest, &selected)
        .is_err());
    let artifact = manifest.files.values().next().unwrap().clone();
    for path in [
        "../outside",
        "C:/secret",
        "/absolute",
        "x\\secret",
        "nul.bin",
        "a/COM1.txt",
        "a./x",
    ] {
        manifest.files.insert(path.into(), artifact.clone());
        assert!(manifest.validate().is_err());
        manifest.files.remove(path);
    }
}
#[test]
fn ir_pair_strips_each_suffix_once() {
    let mut manifest: Manifest =
        serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    let artifact = manifest.files.values().next().unwrap().clone();
    manifest
        .files
        .insert("u/a.xml.xml".into(), artifact.clone());
    manifest.files.insert("u/a.bin".into(), artifact);
    let unet = manifest.components.get_mut("unet").unwrap();
    unet.xml = "u/a.xml.xml".into();
    unet.weights = "u/a.bin".into();
    rejected_with(manifest.validate(), "IR XML and BIN must share a basename");
}
fn rejected_with(result: cascadia_diffusion::Result<impl std::fmt::Debug>, message: &str) {
    match result {
        Err(cascadia_diffusion::Error::Invalid(m)) => assert_eq!(m, message),
        other => panic!("expected {message:?}, got {other:?}"),
    }
}
#[test]
fn f16_infinity_and_nan_are_not_finite() {
    assert!(Dtype::F16.finite(&[0x00, 0x3c]));
    for bits in [0x7c00u16, 0xfc00, 0x7e00] {
        assert!(!Dtype::F16.finite(&bits.to_le_bytes()));
    }
}
#[test]
fn header_longer_than_max_header_is_rejected_even_when_well_formed() {
    let bytes = include_bytes!("fixtures/frame.bin");
    let n = u32::from_be_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let (header, payload) = (&bytes[8..8 + n], &bytes[8 + n..]);
    // Trailing JSON whitespace keeps the header valid, so only the cap rejects it.
    let mut padded = header.to_vec();
    padded.resize(wire::MAX_HEADER + 1, b' ');
    let mut frame = b"CDF1".to_vec();
    frame.extend_from_slice(&(padded.len() as u32).to_be_bytes());
    frame.extend_from_slice(&padded);
    frame.extend_from_slice(payload);
    rejected_with(wire::decode(&frame), "invalid header length");
}
/// A verified copy of the fixture bundle.
fn bundle() -> (tempfile::TempDir, Manifest, String) {
    let manifest: Manifest = serde_json::from_str(include_str!("fixtures/manifest.json")).unwrap();
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("unet")).unwrap();
    std::fs::write(root.path().join("unet/model.xml"), b"xml").unwrap();
    std::fs::write(root.path().join("unet/model.bin"), b"bin").unwrap();
    let digest = manifest.id().unwrap();
    manifest
        .verify_bundle(root.path(), &digest, &["unet".into()])
        .unwrap();
    (root, manifest, digest)
}
#[test]
fn unlisted_file_in_bundle_is_rejected() {
    let (root, manifest, digest) = bundle();
    std::fs::write(root.path().join("unet/extra.json"), b"{}").unwrap();
    rejected_with(
        manifest.verify_bundle(root.path(), &digest, &["unet".into()]),
        "unlisted artifact",
    );
}
#[cfg(unix)]
#[test]
fn symlinks_and_nonregular_entries_in_bundle_are_rejected() {
    use std::os::unix::fs::symlink;
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("model.bin"), b"bin").unwrap();
    // A listed artifact that is a link to identical bytes.
    let (root, manifest, digest) = bundle();
    std::fs::remove_file(root.path().join("unet/model.bin")).unwrap();
    symlink(
        outside.path().join("model.bin"),
        root.path().join("unet/model.bin"),
    )
    .unwrap();
    rejected_with(
        manifest.verify_bundle(root.path(), &digest, &["unet".into()]),
        "artifact symlink",
    );
    // An unlisted link elsewhere in the bundle.
    let (root, manifest, digest) = bundle();
    symlink(outside.path(), root.path().join("linked")).unwrap();
    rejected_with(
        manifest.verify_bundle(root.path(), &digest, &["unet".into()]),
        "bundle symlink",
    );
    let (root, manifest, digest) = bundle();
    let made = std::process::Command::new("mkfifo")
        .arg(root.path().join("pipe"))
        .status()
        .unwrap();
    assert!(made.success());
    rejected_with(
        manifest.verify_bundle(root.path(), &digest, &["unet".into()]),
        "nonregular artifact",
    );
}
