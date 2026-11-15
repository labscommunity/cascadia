//! Camera slot order: which camera the model sees in which `images` slot.
//!
//! Never taken from the client. The first rule that applies wins:
//! 1. `smolvla_resize` with a non-empty `image_key_reorder_map`: each camera
//!    at its mapped slot (`smolvla.py` `_camera_slot_layout`).
//! 2. VISUAL `input_features` in manifest order (`model.py` `_order_cameras`).
//! 3. Neither: `camera_0..camera_{n-1}` in IR slot order, `n` from the IR.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::manifest::{FeatureType, ImageStep, PolicyManifest};
use super::steps::image_to_chw;

/// Advertised camera names and, per model slot, the camera in it (`None`: a masked dummy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraLayout {
    pub names: Vec<String>,
    pub slots: Vec<Option<usize>>,
}

/// What the manifest settles about cameras before the IR is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CameraPlan {
    Fixed(CameraLayout),
    /// Rule 3: the IR's slot count minus `dummies` real cameras, then the dummies.
    FromIr {
        dummies: usize,
    },
}

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!("policy cameras: {msg}")))
}

/// `images` is the bare single-camera key; `images.<name>` advertises `<name>`.
fn advertised(key: &str) -> Option<String> {
    match key.strip_prefix("images") {
        Some("") => Some("image".into()),
        Some(rest) => rest
            .strip_prefix('.')
            .filter(|n| !n.is_empty())
            .map(str::to_string),
        None => None,
    }
}

impl CameraPlan {
    pub fn from_manifest(m: &PolicyManifest) -> EngineResult<Self> {
        let (empty_cameras, smolvla) = match m.image_step() {
            ImageStep::Pi05(p) => (p.empty_cameras, None),
            ImageStep::SmolVlaResize(s) => (0, Some(s)),
        };

        if let Some(s) = smolvla.filter(|s| !s.image_key_reorder_map.is_empty()) {
            let mut entries = Vec::new();
            for (key, slot) in &s.image_key_reorder_map {
                if *slot < 0 {
                    return refuse(format!("image_key_reorder_map[{key:?}] is negative"));
                }
                // smolvla.py `_normalize_image_key` prefixes bare names with `images.`.
                let name = advertised(key).unwrap_or_else(|| key.clone());
                if entries.iter().any(|(n, _)| *n == name) {
                    return refuse(format!("camera {name:?} mapped twice"));
                }
                if entries.iter().any(|(_, s)| *s as i64 == *slot) {
                    return refuse(format!("image_key_reorder_map slot {slot} used twice"));
                }
                entries.push((name, *slot as usize));
            }
            entries.sort_by_key(|(_, slot)| *slot);
            let names: Vec<String> = entries.iter().map(|(n, _)| n.clone()).collect();
            if s.num_cameras <= 0 {
                return Ok(Self::Fixed(CameraLayout {
                    slots: (0..names.len()).map(Some).collect(),
                    names,
                }));
            }
            let num = s.num_cameras as usize;
            let mut slots = vec![None; num];
            for (i, (name, slot)) in entries.iter().enumerate() {
                if *slot >= num {
                    return refuse(format!(
                        "camera {name:?} maps to slot {slot}, but num_cameras is {num}"
                    ));
                }
                slots[*slot] = Some(i);
            }
            return Ok(Self::Fixed(CameraLayout { names, slots }));
        }

        let visual: Vec<&str> = m
            .input_features
            .iter()
            .filter(|f| f.ftype == FeatureType::Visual)
            .map(|f| f.name.as_str())
            .collect();
        // With no map, smolvla fills slots in camera order, then dummies up to num_cameras.
        let num_cameras = smolvla.map_or(0, |s| s.num_cameras);

        if visual.is_empty() {
            if num_cameras > 0 {
                return refuse(
                    "num_cameras without input_features or image_key_reorder_map \
                     leaves the real camera count unknown",
                );
            }
            return Ok(Self::FromIr {
                dummies: empty_cameras,
            });
        }

        let mut names: Vec<String> = Vec::new();
        for key in &visual {
            let Some(name) = advertised(key) else {
                return refuse(format!(
                    "visual feature {key:?} is neither `images` nor `images.<name>`"
                ));
            };
            if names.contains(&name) {
                return refuse(format!("camera {name:?} declared twice"));
            }
            names.push(name);
        }
        if visual.len() > 1 && visual.contains(&"images") {
            return refuse("a bare `images` feature next to named cameras");
        }
        let dummies = if num_cameras > 0 {
            let num = num_cameras as usize;
            if names.len() > num {
                return refuse(format!(
                    "{} cameras do not fit num_cameras {num}",
                    names.len()
                ));
            }
            num - names.len()
        } else {
            empty_cameras
        };
        let slots = (0..names.len())
            .map(Some)
            .chain(std::iter::repeat_n(None, dummies))
            .collect();
        Ok(Self::Fixed(CameraLayout { names, slots }))
    }

    /// Settle the layout against the IR's `images` slot count (its first dimension).
    pub fn resolve(self, ir_slots: usize) -> EngineResult<CameraLayout> {
        match self {
            Self::Fixed(l) if l.slots.len() == ir_slots => Ok(l),
            Self::Fixed(l) => refuse(format!(
                "the manifest gives {} image slots, the model has {ir_slots}",
                l.slots.len()
            )),
            Self::FromIr { dummies } if ir_slots > dummies => {
                let n = ir_slots - dummies;
                Ok(CameraLayout {
                    names: (0..n).map(|i| format!("camera_{i}")).collect(),
                    slots: (0..n)
                        .map(Some)
                        .chain(std::iter::repeat_n(None, dummies))
                        .collect(),
                })
            }
            Self::FromIr { dummies } => refuse(format!(
                "the model has {ir_slots} image slots, not more than the {dummies} empty ones"
            )),
        }
    }
}

/// Build `images [n_slots, 1, 3, H, W]` f32 and `image_masks [n_slots, 1]`.
///
/// Dummy slots are all `-1` with mask false (`pi05.py` `empty_cameras`,
/// `smolvla.py` unassigned slots). Frames must already be `height x width`.
pub fn images_tensor(
    layout: &CameraLayout,
    images: &[(String, Image)],
    height: u32,
    width: u32,
) -> EngineResult<(Vec<f32>, Vec<bool>)> {
    let bad = |msg: String| Err(EngineError::InvalidConfig(msg));
    if images.len() != layout.names.len() {
        return bad(format!(
            "{} cameras sent; expected {:?}",
            images.len(),
            layout.names
        ));
    }
    let (h, w) = (height as usize, width as usize);
    let per_slot = 3 * h * w;
    let mut data = Vec::with_capacity(layout.slots.len() * per_slot);
    let mut masks = Vec::with_capacity(layout.slots.len());
    for slot in &layout.slots {
        let Some(i) = slot else {
            data.extend(std::iter::repeat_n(-1.0f32, per_slot));
            masks.push(false);
            continue;
        };
        let name = &layout.names[*i];
        let Some((_, img)) = images.iter().find(|(n, _)| n == name) else {
            return bad(format!("missing camera {name:?}"));
        };
        if img.height != height || img.width != width || img.data.len() != per_slot {
            return bad(format!(
                "camera {name:?} must be {height}x{width} RGB ({per_slot} bytes)"
            ));
        }
        image_to_chw(&img.data, h, w, &mut data);
        masks.push(true);
    }
    Ok((data, masks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::manifest::LoadOptions;
    use serde_json::{json, Value};

    const PI05_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-pi05-libero-manifest.json");
    const SMOLVLA_LIBERO: &str =
        include_str!("../../tests/fixtures/policy/intel-smolvla-libero-manifest.json");

    fn manifest(image_step: Value, visual: &[&str]) -> PolicyManifest {
        let features: Vec<Value> = visual
            .iter()
            .map(|n| {
                json!({"class_path": "physicalai.inference.data.features.InferenceFeature",
                       "init_args": {"ftype": "VISUAL", "name": n, "shape": [3, 8, 8], "dtype": "float32"}})
            })
            .collect();
        let v = json!({
            "format": "policy_package", "version": "1.0", "policy": {"name": "smolvla"},
            "model": {
                "artifacts": {"openvino": "m.xml"},
                "preprocessors": [image_step, {"type": "ov_tokenizer", "artifact": "t.xml"}],
                "input_features": features
            }
        });
        PolicyManifest::parse(&v.to_string(), LoadOptions::default()).unwrap()
    }

    fn plan(image_step: Value, visual: &[&str]) -> EngineResult<CameraPlan> {
        CameraPlan::from_manifest(&manifest(image_step, visual))
    }

    fn fixed(names: &[&str], slots: &[Option<usize>]) -> CameraPlan {
        CameraPlan::Fixed(CameraLayout {
            names: names.iter().map(|s| s.to_string()).collect(),
            slots: slots.to_vec(),
        })
    }

    #[test]
    fn rule1_gapped_map_with_num_cameras_places_dummies() {
        let p = plan(
            json!({"type": "smolvla_resize",
                   "image_key_reorder_map": {"images.wrist": 2, "top": 0}, "num_cameras": 3}),
            &["images.ignored"],
        )
        .unwrap();
        assert_eq!(p, fixed(&["top", "wrist"], &[Some(0), None, Some(1)]));
    }

    #[test]
    fn rule1_num_cameras_zero_orders_by_slot_value_without_dummies() {
        // smolvla.py:255: num_cameras <= 0 sorts by slot value; gaps collapse.
        let p = plan(
            json!({"type": "smolvla_resize",
                   "image_key_reorder_map": {"a": 5, "b": 1, "c": 3}}),
            &[],
        )
        .unwrap();
        assert_eq!(p, fixed(&["b", "c", "a"], &[Some(0), Some(1), Some(2)]));
        // A bare `images` key advertises `image`.
        let p = plan(
            json!({"type": "smolvla_resize", "image_key_reorder_map": {"images": 0}}),
            &[],
        )
        .unwrap();
        assert_eq!(p, fixed(&["image"], &[Some(0)]));
    }

    #[test]
    fn rule1_refusals() {
        let map = |m: Value, n: i64| json!({"type": "smolvla_resize", "image_key_reorder_map": m, "num_cameras": n});
        let e = |m, n| plan(map(m, n), &[]).unwrap_err().to_string();
        assert!(e(json!({"a": 0, "b": 0}), 0).contains("used twice"));
        assert!(e(json!({"a": -1}), 0).contains("negative"));
        assert!(e(json!({"a": 0, "b": 3}), 3).contains("slot 3, but num_cameras is 3"));
        assert!(e(json!({"top": 0, "images.top": 1}), 0).contains("mapped twice"));
    }

    #[test]
    fn rule2_visual_features_in_manifest_order() {
        let m = PolicyManifest::parse(PI05_LIBERO, LoadOptions::default()).unwrap();
        // Intel's π0.5 LIBERO: two cameras plus one empty_cameras dummy.
        let p = CameraPlan::from_manifest(&m).unwrap();
        assert_eq!(p, fixed(&["image", "image2"], &[Some(0), Some(1), None]));
        assert_eq!(p.clone().resolve(3).unwrap().names, vec!["image", "image2"]);
        assert!(p
            .resolve(2)
            .unwrap_err()
            .to_string()
            .contains("3 image slots"));
        // An empty map falls through to rule 2.
        let p = plan(
            json!({"type": "smolvla_resize", "image_key_reorder_map": {}}),
            &["images.wrist", "images.top"],
        )
        .unwrap();
        assert_eq!(p, fixed(&["wrist", "top"], &[Some(0), Some(1)]));
        // Without a map, smolvla pads to num_cameras after the real cameras.
        let p = plan(
            json!({"type": "smolvla_resize", "num_cameras": 3}),
            &["images.wrist"],
        )
        .unwrap();
        assert_eq!(p, fixed(&["wrist"], &[Some(0), None, None]));
    }

    #[test]
    fn rule2_single_bare_images_feature_is_image() {
        let p = plan(json!({"type": "pi05"}), &["images"]).unwrap();
        assert_eq!(p, fixed(&["image"], &[Some(0)]));
        let e = plan(json!({"type": "pi05"}), &["images", "images.top"]).unwrap_err();
        assert!(e.to_string().contains("bare `images`"));
        let e = plan(json!({"type": "pi05"}), &["observation.images.top"]).unwrap_err();
        assert!(e.to_string().contains("neither"));
        let e = plan(json!({"type": "pi05"}), &["images.a", "images.a"]).unwrap_err();
        assert!(e.to_string().contains("declared twice"));
    }

    #[test]
    fn rule3_names_cameras_from_the_ir() {
        let m = PolicyManifest::parse(SMOLVLA_LIBERO, LoadOptions::default()).unwrap();
        let p = CameraPlan::from_manifest(&m).unwrap();
        assert_eq!(p, CameraPlan::FromIr { dummies: 0 });
        let l = p.resolve(2).unwrap();
        assert_eq!(l.names, vec!["camera_0", "camera_1"]);
        assert_eq!(l.slots, vec![Some(0), Some(1)]);
        let p = plan(json!({"type": "pi05", "empty_cameras": 1}), &[]).unwrap();
        assert_eq!(
            p.clone().resolve(3).unwrap().slots,
            vec![Some(0), Some(1), None]
        );
        assert!(p.resolve(1).is_err());
        let e = plan(json!({"type": "smolvla_resize", "num_cameras": 2}), &[]).unwrap_err();
        assert!(e.to_string().contains("real camera count unknown"));
    }

    #[test]
    fn images_tensor_fills_slots_and_dummies() {
        let layout = CameraLayout {
            names: vec!["top".into(), "wrist".into()],
            slots: vec![Some(1), None, Some(0)],
        };
        let img = |v: u8| Image {
            height: 1,
            width: 1,
            data: vec![v, v, v],
        };
        let images = vec![("top".to_string(), img(255)), ("wrist".to_string(), img(0))];
        let (data, masks) = images_tensor(&layout, &images, 1, 1).unwrap();
        assert_eq!(
            data,
            vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(masks, vec![true, false, true]);
        assert!(images_tensor(&layout, &images[..1], 1, 1).is_err());
        let wrong = vec![("top".to_string(), img(1)), ("side".to_string(), img(1))];
        assert!(images_tensor(&layout, &wrong, 1, 1)
            .unwrap_err()
            .to_string()
            .contains("missing camera \"wrist\""));
        assert!(images_tensor(&layout, &images, 2, 1).is_err());
    }
}
