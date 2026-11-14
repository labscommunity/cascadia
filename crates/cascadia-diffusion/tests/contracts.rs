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
