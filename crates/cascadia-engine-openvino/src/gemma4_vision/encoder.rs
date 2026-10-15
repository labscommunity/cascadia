//! The Gemma 4 vision IR (`openvino_vision_embeddings_model.xml`): padded
//! patches and their positions in, one `hidden_size` row per soft token out
//! (the IR pools `pooling²` patches per token and drops the padding).

use cascadia_ov_genai_shim::{DType, PluginConfig, Runtime};

use super::Patches;

pub struct VisionEncoder {
    rt: Runtime,
    pooling_kernel_size: u32,
}

impl VisionEncoder {
    /// Compile the vision IR from `model_dir` on `device`.
    pub fn load(
        model_dir: &std::path::Path,
        device: &str,
        plugin: &PluginConfig,
        pooling_kernel_size: u32,
    ) -> Result<Self, String> {
        let path = model_dir.join("openvino_vision_embeddings_model.xml");
        let rt = Runtime::compile(&path.to_string_lossy(), device, plugin)
            .map_err(|e| format!("compiling {}: {e}", path.display()))?;
        Ok(Self {
            rt,
            pooling_kernel_size,
        })
    }

    /// Image embeddings, row-major `[soft_tokens, hidden]`, and `hidden`.
    pub fn encode(&mut self, p: &Patches) -> Result<(Vec<f32>, usize), String> {
        let pixels: Vec<u8> = p
            .pixel_values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let positions: Vec<u8> = p
            .position_ids
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        self.rt
            .set_input("pixel_values", DType::F32, &[1, p.rows, p.cols], &pixels)
            .and_then(|_| {
                self.rt.set_input(
                    "image_position_ids",
                    DType::I64,
                    &[1, p.rows, 2],
                    &positions,
                )
            })
            .and_then(|_| self.rt.infer())
            .map_err(|e| format!("vision embedding: {e}"))?;
        let (dtype, shape, bytes) = self
            .rt
            .output(0)
            .map_err(|e| format!("vision embedding: {e}"))?;
        if dtype != DType::F32 {
            return Err(format!(
                "vision embedding output is {dtype:?}, expected f32"
            ));
        }
        let want = p.soft_tokens(self.pooling_kernel_size);
        // [tokens, hidden], tolerating a leading batch dimension of 1.
        let (tokens, hidden) = match shape[..] {
            [1, t, h] | [t, h] => (t, h),
            _ => return Err(format!("vision embedding returned shape {shape:?}")),
        };
        if tokens != want {
            return Err(format!(
                "vision embedding returned {tokens} tokens, expected {want} ({} patches)",
                p.num_patches
            ));
        }
        let v = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((v, hidden))
    }
}
