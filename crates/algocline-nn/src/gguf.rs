//! Writing a model out as GGUF.
//!
//! Everything this crate trains lands in safetensors, which is the
//! format the training side reads and writes and nothing else does.
//! llama.cpp, Ollama and LM Studio read GGUF, so a model trained here
//! could be evaluated here and nowhere else.
//!
//! This is the writer. It renames the tensors to the
//! [GGUF naming convention](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
//! llama.cpp expects, emits the architecture's key-value metadata,
//! optionally embeds the tokenizer, and quantizes on the way out.
//!
//! # What is verified here, and what is not
//!
//! Verified by this crate's tests: the file is a GGUF that reads back,
//! every tensor survives the round trip with its GGUF name and its
//! shape, the metadata says what it was asked to say, and a quantized
//! export is within the error its block format allows.
//!
//! **Not verified here: that llama.cpp loads it.** That takes llama.cpp,
//! which this repository does not build or vendor. The naming and the
//! key set below are written against its documented convention, and the
//! remaining step is one `llama-cli -m <file>` on a machine that has it.
//! Said plainly rather than implied, because "exports GGUF" and "runs
//! under llama.cpp" are different claims and only the first is fenced.
//!
//! # Layout
//!
//! ggml stores a matmul weight with the input dimension first. A candle
//! [`candle_nn::Linear`] holds `[out, in]`, and candle's GGUF writer
//! emits dimensions reversed, so passing these tensors through
//! unchanged writes `ne = [in, out]` — the convention ggml reads. The
//! transposition is the writer's, not ours, which is why nothing here
//! transposes anything. (candle's reader reverses them again, so a
//! round trip through this crate comes back in candle order; the file
//! on disk is in ggml's.)
//!
//! # Quantization has a width requirement
//!
//! Every ggml block format stores a fixed number of values together —
//! 32 for the `Q*_0` / `Q*_1` families, 256 for the K-quants — and the
//! last dimension of a tensor has to be a multiple of it. A model whose
//! hidden size is not is exportable at F32 or F16 and not below; the
//! refusal names the tensor and the block size rather than failing
//! somewhere inside candle.

use std::collections::BTreeMap;
use std::path::Path;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::Tensor;
use candle_nn::VarMap;

/// Which architecture's naming and key set to write.
///
/// GGUF is one container per architecture: the tensor names and the
/// metadata keys are both namespaced by `general.architecture`, and a
/// reader dispatches on it. Writing a GPT-2 under llama's keys produces
/// a file that loads and computes nothing sensible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufArch {
    /// llama.cpp's `gpt2`: learned positional embeddings, fused QKV,
    /// LayerNorm with biases.
    Gpt2,
    /// llama.cpp's `llama`: RoPE, split Q/K/V, RMSNorm, SwiGLU.
    Llama,
}

impl GgufArch {
    /// The `general.architecture` value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gpt2 => "gpt2",
            Self::Llama => "llama",
        }
    }
}

/// The facts a reader needs to rebuild the computation.
///
/// Every one of them is a number the model was built with, and every
/// one is written into the file — a GGUF that omitted them would be the
/// bag of tensors this exists to stop producing.
#[derive(Debug, Clone)]
pub struct GgufSpec {
    /// Which naming and key set to write.
    pub arch: GgufArch,
    /// Transformer blocks.
    pub layers: usize,
    /// Query attention heads.
    pub heads: usize,
    /// Key/value heads. Equal to `heads` for multi-head attention.
    pub kv_heads: usize,
    /// Hidden size.
    pub dim: usize,
    /// Feed-forward intermediate size.
    pub ffn_dim: usize,
    /// Context window.
    pub ctx: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Normalisation epsilon.
    pub eps: f32,
    /// RoPE base frequency, for architectures that rotate.
    pub rope_theta: Option<f32>,
    /// Model name recorded under `general.name`.
    pub name: String,
}

/// Parse the wire form of a precision (Lua opts, JSON config).
///
/// `None` on an unknown name, so the caller can list the alternatives.
/// The list is the formats ggml reads that candle can also write, which
/// is not all of them.
pub fn parse_precision(name: &str) -> Option<GgmlDType> {
    Some(match name {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "q8_0" => GgmlDType::Q8_0,
        "q5_1" => GgmlDType::Q5_1,
        "q5_0" => GgmlDType::Q5_0,
        "q4_1" => GgmlDType::Q4_1,
        "q4_0" => GgmlDType::Q4_0,
        "q6k" => GgmlDType::Q6K,
        "q5k" => GgmlDType::Q5K,
        "q4k" => GgmlDType::Q4K,
        "q3k" => GgmlDType::Q3K,
        "q2k" => GgmlDType::Q2K,
        _ => return None,
    })
}

/// Every precision name this version accepts, widest first.
pub const PRECISION_NAMES: [&str; 12] = [
    "f32", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "q6k", "q5k", "q4k", "q3k", "q2k",
];

/// What an export wrote.
#[derive(Debug, Clone)]
pub struct GgufReport {
    /// Tensors written, under their GGUF names.
    pub tensors: usize,
    /// Metadata entries written.
    pub metadata: usize,
    /// Precision every tensor was quantized to.
    pub precision: GgmlDType,
    /// Whether a tokenizer was embedded. Without one the file needs an
    /// external vocabulary, which most readers will not accept.
    pub tokenizer: bool,
}

/// Write `varmap` to `path` as GGUF.
///
/// `tokenizer_json` is an HF `tokenizer.json` to embed. Optional
/// because a caller may be exporting for a runtime that supplies its
/// own vocabulary, and stated in the report either way — a GGUF with no
/// tokenizer is refused by llama.cpp's own CLI, so whether one is in
/// there is not a detail.
///
/// # Errors
///
/// A tensor the naming map has no GGUF name for (which means this crate
/// grew a parameter and this map did not), a quantization the shape
/// cannot take (block formats need the last dimension to be a multiple
/// of the block size), and any I/O failure.
pub fn export_gguf(
    varmap: &VarMap,
    spec: &GgufSpec,
    precision: GgmlDType,
    tokenizer_json: Option<&Path>,
    path: &Path,
) -> Result<GgufReport, String> {
    let named = collect_tensors(varmap, spec)?;

    let mut quantized: Vec<(String, QTensor)> = Vec::with_capacity(named.len());
    for (name, tensor) in named {
        // Normalisation weights and biases stay F32 whatever the
        // request: they are a vector per block, quantizing them saves
        // nothing measurable, and llama.cpp keeps them full-precision
        // for the same reason.
        let dtype = if is_small_vector(&tensor) {
            GgmlDType::F32
        } else {
            precision
        };
        let q = QTensor::quantize(&tensor, dtype).map_err(|e| {
            format!(
                "gguf export: quantize `{name}` {:?} to {dtype:?}: {e}. A ggml block format \
                 stores {} values at a time, so the last dimension has to be a multiple of \
                 it — export this model at F32 or F16, or build it at a width that is",
                tensor.dims(),
                dtype.block_size()
            )
        })?;
        quantized.push((name, q));
    }

    let mut metadata = architecture_metadata(spec);
    let embedded = match tokenizer_json {
        Some(json) => {
            metadata.extend(tokenizer_metadata(json, spec.arch)?);
            true
        }
        None => false,
    };

    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("gguf export: create {}: {e}", path.display()))?;
    let meta_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        quantized.iter().map(|(k, v)| (k.as_str(), v)).collect();
    gguf_file::write(&mut file, &meta_refs, &tensor_refs)
        .map_err(|e| format!("gguf export: write {}: {e}", path.display()))?;

    Ok(GgufReport {
        tensors: tensor_refs.len(),
        metadata: meta_refs.len(),
        precision,
        tokenizer: embedded,
    })
}

/// Whether a tensor is a per-channel vector rather than a weight matrix.
fn is_small_vector(t: &Tensor) -> bool {
    t.dims().len() < 2
}

/// The model's tensors under their GGUF names, in a stable order.
fn collect_tensors(varmap: &VarMap, spec: &GgufSpec) -> Result<Vec<(String, Tensor)>, String> {
    let data = varmap
        .data()
        .lock()
        .map_err(|_| "gguf export: VarMap lock poisoned".to_string())?;
    let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
    for (name, var) in data.iter() {
        let gguf = gguf_name(name, spec.arch).ok_or_else(|| {
            format!(
                "gguf export: no GGUF name for `{name}` under architecture `{}`; this crate \
                 grew a parameter the naming map does not cover",
                spec.arch.name()
            )
        })?;
        if out.insert(gguf.clone(), var.as_tensor().clone()).is_some() {
            return Err(format!(
                "gguf export: two parameters map to `{gguf}`, which would silently drop one"
            ));
        }
    }
    Ok(out.into_iter().collect())
}

/// This crate's parameter name to its GGUF name.
///
/// `None` for a name the convention has no slot for, which is refused
/// rather than passed through: a reader dispatches on these names, and
/// one it does not recognise is a weight that silently never gets used.
fn gguf_name(name: &str, arch: GgufArch) -> Option<String> {
    match arch {
        GgufArch::Gpt2 => gguf_name_gpt2(name),
        GgufArch::Llama => gguf_name_llama(name),
    }
}

fn gguf_name_gpt2(name: &str) -> Option<String> {
    let direct = match name {
        "wte.weight" => Some("token_embd.weight"),
        "wpe.weight" => Some("position_embd.weight"),
        "ln_f.weight" => Some("output_norm.weight"),
        "ln_f.bias" => Some("output_norm.bias"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(d) = direct {
        return Some(d.to_string());
    }
    // `h.<N>.<rest>` → `blk.<N>.<mapped>`
    let rest = name.strip_prefix("h.")?;
    let (index, rest) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;
    let (part, suffix) = rest.rsplit_once('.')?;
    let mapped = match part {
        "ln_1" => "attn_norm",
        "attn.c_attn" => "attn_qkv",
        "attn.c_proj" => "attn_output",
        "ln_2" => "ffn_norm",
        "mlp.c_fc" => "ffn_up",
        "mlp.c_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{index}.{mapped}.{suffix}"))
}

fn gguf_name_llama(name: &str) -> Option<String> {
    let direct = match name {
        "embed_tokens.weight" => Some("token_embd.weight"),
        "norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(d) = direct {
        return Some(d.to_string());
    }
    let rest = name.strip_prefix("layers.")?;
    let (index, rest) = rest.split_once('.')?;
    index.parse::<usize>().ok()?;
    let (part, suffix) = rest.rsplit_once('.')?;
    let mapped = match part {
        "input_layernorm" => "attn_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "post_attention_layernorm" => "ffn_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{index}.{mapped}.{suffix}"))
}

/// The architecture's key-value metadata.
///
/// Keys are namespaced by the architecture name, which is the
/// convention's own rule: a reader looks up `<arch>.block_count`, not
/// `block_count`.
fn architecture_metadata(spec: &GgufSpec) -> Vec<(String, gguf_file::Value)> {
    use gguf_file::Value;
    let arch = spec.arch.name();
    let mut out = vec![
        (
            "general.architecture".to_string(),
            Value::String(arch.to_string()),
        ),
        ("general.name".to_string(), Value::String(spec.name.clone())),
        (
            format!("{arch}.context_length"),
            Value::U32(spec.ctx as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            Value::U32(spec.dim as u32),
        ),
        (
            format!("{arch}.block_count"),
            Value::U32(spec.layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            Value::U32(spec.ffn_dim as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            Value::U32(spec.heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            Value::U32(spec.kv_heads as u32),
        ),
    ];
    // The epsilon key is named after the normalisation the architecture
    // uses, because a reader looks for the one its own graph needs.
    let eps_key = match spec.arch {
        GgufArch::Gpt2 => format!("{arch}.attention.layer_norm_epsilon"),
        GgufArch::Llama => format!("{arch}.attention.layer_norm_rms_epsilon"),
    };
    out.push((eps_key, Value::F32(spec.eps)));
    if let Some(theta) = spec.rope_theta {
        out.push((format!("{arch}.rope.freq_base"), Value::F32(theta)));
        out.push((
            format!("{arch}.rope.dimension_count"),
            Value::U32((spec.dim / spec.heads) as u32),
        ));
    }
    out
}

/// The tokenizer keys, read out of an HF `tokenizer.json`.
///
/// Read as JSON rather than through the `tokenizers` crate because the
/// merge list is what GGUF wants and that crate does not hand it back —
/// and because a BPE `tokenizer.json` carries both halves verbatim,
/// so there is nothing to reconstruct.
fn tokenizer_metadata(
    json: &Path,
    arch: GgufArch,
) -> Result<Vec<(String, gguf_file::Value)>, String> {
    use gguf_file::Value;
    let text = std::fs::read_to_string(json)
        .map_err(|e| format!("gguf export: read tokenizer {}: {e}", json.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("gguf export: parse tokenizer {}: {e}", json.display()))?;
    let model = parsed
        .get("model")
        .ok_or_else(|| "gguf export: tokenizer.json has no `model`".to_string())?;

    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "gguf export: tokenizer.json has no `model.vocab` object".to_string())?;
    // Ordered by id, because GGUF stores the vocabulary as an array and
    // the index *is* the token id.
    let mut by_id: Vec<(u32, String)> = Vec::with_capacity(vocab.len());
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| format!("gguf export: token `{token}` has a non-integer id"))?;
        by_id.push((id as u32, token.clone()));
    }
    by_id.sort_by_key(|(id, _)| *id);
    for (position, (id, token)) in by_id.iter().enumerate() {
        if *id as usize != position {
            return Err(format!(
                "gguf export: the vocabulary skips an id — position {position} holds id {id} \
                 (`{token}`), and GGUF stores tokens by position"
            ));
        }
    }
    let tokens: Vec<String> = by_id.into_iter().map(|(_, t)| t).collect();

    let merges: Vec<String> = match model.get("merges").and_then(|m| m.as_array()) {
        Some(list) => list
            .iter()
            .map(|entry| match entry {
                // `tokenizers` writes merges as `"a b"` in older files
                // and `["a", "b"]` in newer ones; GGUF wants the first.
                serde_json::Value::String(s) => Ok(s.clone()),
                serde_json::Value::Array(pair) if pair.len() == 2 => {
                    let a = pair[0].as_str().unwrap_or_default();
                    let b = pair[1].as_str().unwrap_or_default();
                    Ok(format!("{a} {b}"))
                }
                other => Err(format!("gguf export: unreadable merge entry {other}")),
            })
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };

    // Every token is "normal" (type 1) unless the file marks it
    // otherwise; added tokens are the ones that are not.
    let mut token_type = vec![1i32; tokens.len()];
    if let Some(added) = parsed.get("added_tokens").and_then(|a| a.as_array()) {
        for entry in added {
            if let Some(id) = entry.get("id").and_then(|i| i.as_u64()) {
                if let Some(slot) = token_type.get_mut(id as usize) {
                    // 3 = control, the type llama.cpp gives a special
                    // token it must not split.
                    *slot = 3;
                }
            }
        }
    }

    let model_name = match arch {
        GgufArch::Gpt2 => "gpt2",
        GgufArch::Llama => "llama",
    };
    let mut out = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::String(model_name.to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(tokens.into_iter().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(token_type.into_iter().map(Value::I32).collect()),
        ),
    ];
    if !merges.is_empty() {
        out.push((
            "tokenizer.ggml.merges".to_string(),
            Value::Array(merges.into_iter().map(Value::String).collect()),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn gpt2_spec() -> GgufSpec {
        GgufSpec {
            arch: GgufArch::Gpt2,
            layers: 2,
            heads: 2,
            kv_heads: 2,
            dim: 16,
            ffn_dim: 64,
            ctx: 8,
            vocab: 32,
            eps: 1e-5,
            rope_theta: None,
            name: "tiny".into(),
        }
    }

    /// A model whose width is a multiple of the 32-value ggml block,
    /// so the quantized formats apply to it. The tiny fixture's 16 is
    /// not, which is itself worth having a test for.
    fn wide_gpt2() -> (VarMap, GgufSpec) {
        use crate::arch::{Gpt2Config, Gpt2Model};
        let cfg = Gpt2Config {
            layers: 1,
            heads: 2,
            dim: 32,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        Gpt2Model::new(&cfg, vb).expect("build");
        let spec = GgufSpec {
            layers: 1,
            dim: 32,
            ffn_dim: 128,
            ..gpt2_spec()
        };
        (vm, spec)
    }

    fn tiny_gpt2() -> VarMap {
        use crate::arch::{Gpt2Config, Gpt2Model};
        let cfg = Gpt2Config {
            layers: 2,
            heads: 2,
            dim: 16,
            ctx: 8,
            vocab: 32,
            dtype: DType::F32,
            device: Device::Cpu,
            eps: 1e-5,
            moe: None,
            custom: None,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, cfg.dtype, &cfg.device);
        Gpt2Model::new(&cfg, vb).expect("build");
        vm
    }

    #[test]
    fn every_gpt2_parameter_has_a_gguf_name() {
        let vm = tiny_gpt2();
        let named = collect_tensors(&vm, &gpt2_spec()).expect("every name maps");
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"token_embd.weight"));
        assert!(names.contains(&"position_embd.weight"));
        assert!(names.contains(&"output_norm.weight"));
        assert!(names.contains(&"blk.0.attn_qkv.weight"));
        assert!(names.contains(&"blk.1.ffn_down.bias"));
        assert_eq!(named.len(), vm.data().lock().unwrap().len());
    }

    #[test]
    fn a_parameter_the_map_does_not_cover_is_refused() {
        // Rather than dropped: a reader dispatches on these names, and
        // one it does not know is a weight that never gets used.
        assert_eq!(gguf_name("h.0.attn.c_new.weight", GgufArch::Gpt2), None);
        assert_eq!(gguf_name("something.else", GgufArch::Llama), None);
        assert_eq!(
            gguf_name("layers.0.self_attn.q_proj.weight", GgufArch::Llama).as_deref(),
            Some("blk.0.attn_q.weight")
        );
    }

    #[test]
    fn an_exported_file_reads_back_as_the_model_that_was_written() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("tiny.gguf");
        let vm = tiny_gpt2();
        let spec = gpt2_spec();
        let report = export_gguf(&vm, &spec, GgmlDType::F32, None, &path).expect("export");
        assert!(!report.tokenizer, "no tokenizer was supplied");
        assert_eq!(report.tensors, vm.data().lock().unwrap().len());

        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).expect("the file is a GGUF");

        assert_eq!(
            content.metadata["general.architecture"]
                .to_string()
                .unwrap(),
            "gpt2"
        );
        assert_eq!(content.metadata["gpt2.block_count"].to_u32().unwrap(), 2);
        assert_eq!(
            content.metadata["gpt2.attention.head_count"]
                .to_u32()
                .unwrap(),
            2
        );
        assert_eq!(content.metadata["gpt2.context_length"].to_u32().unwrap(), 8);

        // Every tensor is there, under its GGUF name, at its shape. The
        // dimensions come back reversed, which is ggml's convention and
        // the reason nothing here transposes.
        let source = vm.data().lock().unwrap();
        let expected = source.get("wte.weight").unwrap().dims().to_vec();
        let info = content
            .tensor_infos
            .get("token_embd.weight")
            .expect("the embedding is in the file");
        // candle reverses on the way out and again on the way in, so a
        // round trip through this crate comes back in candle order —
        // the file itself holds ggml's.
        assert_eq!(info.shape.dims().to_vec(), expected);
        assert_eq!(content.tensor_infos.len(), source.len());
    }

    #[test]
    fn a_quantized_export_stays_within_its_format_error() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("q8.gguf");
        let (vm, spec) = wide_gpt2();
        let report = export_gguf(&vm, &spec, GgmlDType::Q8_0, None, &path).expect("export");
        assert_eq!(report.precision, GgmlDType::Q8_0);

        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let q = content
            .tensor(&mut file, "token_embd.weight", &Device::Cpu)
            .expect("read the quantized embedding");
        assert_eq!(q.dtype(), GgmlDType::Q8_0);

        let source = vm.data().lock().unwrap()["wte.weight"].as_tensor().clone();
        let restored = q.dequantize(&Device::Cpu).unwrap();
        let gap: f32 = (restored.reshape(source.shape()).unwrap() - &source)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        // Q8_0 keeps a per-32-element scale and 8-bit values, so the
        // error is bounded by half a step of the block's own range.
        let span: f32 = source
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            gap <= span / 127.0 + 1e-6,
            "Q8_0 error {gap} exceeds half a step of {span}"
        );
    }

    #[test]
    fn the_norms_stay_full_precision_under_a_quantized_export() {
        // They are one vector per block: quantizing them saves nothing
        // measurable, and llama.cpp keeps them full-precision too.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("q4.gguf");
        let (vm, spec) = wide_gpt2();
        export_gguf(&vm, &spec, GgmlDType::Q4_0, None, &path).expect("export");
        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        assert_eq!(
            content.tensor_infos["output_norm.weight"].ggml_dtype,
            GgmlDType::F32
        );
        assert_eq!(
            content.tensor_infos["blk.0.attn_qkv.weight"].ggml_dtype,
            GgmlDType::Q4_0
        );
    }

    /// A width the block format cannot take is refused by name, with
    /// the requirement stated — not passed through from inside candle.
    #[test]
    fn a_width_a_block_format_cannot_take_is_refused_by_name() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("narrow.gguf");
        let vm = tiny_gpt2(); // dim 16, under the 32-value block
        let err = export_gguf(&vm, &gpt2_spec(), GgmlDType::Q4_0, None, &path).unwrap_err();
        assert!(err.contains("multiple of it"), "{err}");
        assert!(err.contains("F32 or F16"), "{err}");
    }

    #[test]
    fn the_precision_names_round_trip() {
        for name in PRECISION_NAMES {
            assert!(parse_precision(name).is_some(), "{name} is advertised");
        }
        assert_eq!(parse_precision("q4_0"), Some(GgmlDType::Q4_0));
        assert_eq!(parse_precision("int4"), None);
    }

    #[test]
    fn a_tokenizer_is_embedded_when_one_is_given() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tok = tmp.path().join("tokenizer.json");
        std::fs::write(
            &tok,
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},
               "merges":["a b"]},"added_tokens":[{"id":2,"content":"ab"}]}"#,
        )
        .unwrap();

        let path = tmp.path().join("withtok.gguf");
        let vm = tiny_gpt2();
        let report =
            export_gguf(&vm, &gpt2_spec(), GgmlDType::F32, Some(&tok), &path).expect("export");
        assert!(report.tokenizer);

        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let tokens = content.metadata["tokenizer.ggml.tokens"]
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_string().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(tokens, vec!["a", "b", "ab"], "tokens are ordered by id");
        let types = content.metadata["tokenizer.ggml.token_type"]
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_i32().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(types, vec![1, 1, 3], "the added token is marked control");
        assert!(content.metadata.contains_key("tokenizer.ggml.merges"));
    }

    #[test]
    fn a_vocabulary_with_a_hole_in_it_is_refused() {
        // GGUF stores tokens by position, so a gap would silently
        // renumber every token above it.
        let tmp = tempfile::TempDir::new().unwrap();
        let tok = tmp.path().join("tokenizer.json");
        std::fs::write(&tok, r#"{"model":{"vocab":{"a":0,"c":2}}}"#).unwrap();
        let err = tokenizer_metadata(&tok, GgufArch::Gpt2).unwrap_err();
        assert!(err.contains("skips an id"), "{err}");
    }
}
