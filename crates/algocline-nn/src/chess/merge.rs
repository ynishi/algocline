//! Building a multi-slot checkpoint out of single-slot ones.
//!
//! Plan 10 asks whether the composition plan 09 confirmed has to be
//! *trained*, or whether it can be *assembled* from models that each
//! learned one attribute. This is the assembly.
//!
//! # The merge, and why it is this one
//!
//! Two checkpoints trained on the same corpus, the same recipe and the
//! same seed, differing only in which attribute their band list names.
//! The merged model takes:
//!
//! - the **mean** of every body tensor (`wte`, `wpe`, `h.*`, `ln_f`),
//! - the **concatenation** of the two conditioning tables, in the order
//!   the sources are given.
//!
//! The concatenation is what makes this a merge rather than an
//! interpolation of one thing: each row of `cond_wte` was learned by
//! one model for one attribute, and stacking them keeps every row
//! intact — nothing is averaged that describes different attributes.
//! The body is averaged because it is the part the two runs learned
//! redundantly (the same games, the same objective), which is the
//! premise weight averaging rests on.
//!
//! What this is **not**: a coefficiented interpolation, a task-arithmetic
//! difference, or a merge of models trained on different corpora. Plan
//! 10 measures the simplest form first, and a form that needs a knob is
//! a form whose knob would need registering.
//!
//! # What is checked
//!
//! A merge of two models that disagree on anything the merge cannot
//! reconcile — a different architecture, a different vocabulary, a
//! different context — produces a well-formed file that no shape
//! describes. Those are refused here by name. What is **not** checked
//! is that the two were trained on the same corpus or the same seed:
//! nothing in a checkpoint records it, and the plan's provenance is the
//! operator's.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use thiserror::Error;

use crate::chess::{CondEncoding, ModelShape};

/// The conditioning table's tensor name — the one entry of a
/// checkpoint that is stacked rather than averaged.
const COND_TABLE: &str = "cond_wte.weight";

/// Why a merge could not be built.
#[derive(Debug, Error)]
pub enum MergeError {
    /// A source could not be read, or the output could not be written.
    #[error("{path}: {message}")]
    Io {
        /// Path involved.
        path: String,
        /// Underlying message.
        message: String,
    },

    /// Fewer than two sources were given.
    ///
    /// One source is a copy, not a merge, and answering it here would
    /// let a mis-scripted run produce a "merged" checkpoint that is one
    /// of its inputs.
    #[error("a merge takes at least two sources; {found} given")]
    TooFewSources {
        /// Sources given.
        found: usize,
    },

    /// A source is not a single-slot per-position checkpoint.
    ///
    /// The merge stacks each source's conditioning table as one slot,
    /// so a source that already carries several slots — or that
    /// conditions by prefix, and so has no table to stack — is not a
    /// thing this can assemble.
    #[error(
        "source {path} conditions by {encoding} in {groups} slot(s); a merge takes \
         single-slot per-position checkpoints"
    )]
    NotSingleSlot {
        /// Source involved.
        path: String,
        /// Its conditioning convention.
        encoding: CondEncoding,
        /// Its condition-group count.
        groups: usize,
    },

    /// Two sources disagree on something the merge cannot reconcile.
    ///
    /// The body tensors are averaged elementwise, so a difference in
    /// architecture or vocabulary has no meaning to average; and the
    /// merged shape claims one set of dimensions, which a source that
    /// disagrees would not be described by.
    #[error(
        "sources disagree on {field}: {first} and {other}; the merge averages body tensors \
         elementwise and writes one shape, so there is nothing to reconcile them with"
    )]
    Mismatch {
        /// Which field.
        field: &'static str,
        /// The first source's value.
        first: String,
        /// The disagreeing source's value.
        other: String,
    },

    /// Two sources carry the same band token.
    ///
    /// Each token becomes one row of the merged conditioning table, so
    /// a repeat would leave two rows meaning the same thing and a cell
    /// naming it ambiguous.
    #[error("band {token} appears in more than one source; each token becomes one table row")]
    DuplicateBand {
        /// The repeated token.
        token: String,
    },

    /// A shared base carries a conditioning table of its own.
    ///
    /// [`merge_task_arithmetic`] adds each source's difference from the
    /// base, and the sources' tables are stacked rather than differenced
    /// — a base with a table would either collide with that stacking or
    /// silently decide the merged model's conditioning, which is the one
    /// thing a shared base must not do.
    #[error(
        "the base {path} carries a conditioning table; a shared base must not decide the \
         merged model's conditioning — bake it with prefix conditioning so it carries a body \
         and nothing else"
    )]
    BaseIsConditioned {
        /// The base involved.
        path: String,
    },

    /// A tensor a source carries is missing from another, or the two
    /// disagree on its shape.
    #[error("tensor {name}: {message}")]
    Tensor {
        /// Tensor name.
        name: String,
        /// What went wrong.
        message: String,
    },
}

/// Merge single-slot checkpoints into one multi-slot checkpoint.
///
/// `sources` are checkpoint paths, in the slot order the merged model
/// should carry (the first source's bands become group 0). Writes the
/// merged weights to `out` and its sidecar beside them, and returns the
/// merged shape.
///
/// The sidecar is written through [`ModelShape::save`], so the merged
/// checkpoint is indistinguishable from a trained one to every reader
/// downstream — which is the point: plan 10 scores it with the judge
/// that scored the trained arm.
///
/// # Errors
///
/// See [`MergeError`]. Every refusal is a disagreement the merge cannot
/// reconcile rather than a preference.
pub fn merge_slots(sources: &[&Path], out: &Path) -> Result<ModelShape, MergeError> {
    if sources.len() < 2 {
        return Err(MergeError::TooFewSources {
            found: sources.len(),
        });
    }

    let mut shapes: Vec<ModelShape> = Vec::with_capacity(sources.len());
    for path in sources {
        let shape = ModelShape::load_any(path).map_err(|e| MergeError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let groups = shape.effective_cond_groups().len();
        if shape.encoding != CondEncoding::EveryPosition || groups != 1 {
            return Err(MergeError::NotSingleSlot {
                path: path.display().to_string(),
                encoding: shape.encoding,
                groups,
            });
        }
        shapes.push(shape);
    }

    // Every axis the merged shape claims for all of them. Named one by
    // one rather than compared as whole shapes, so the message says
    // which axis moved — and listed here rather than derived, so a new
    // field on `ModelShape` is a decision someone makes rather than an
    // axis that silently stops being checked.
    let first = &shapes[0];
    for shape in shapes.iter().skip(1) {
        let axes: [(&'static str, String, String); 6] = [
            ("layers", first.layers.to_string(), shape.layers.to_string()),
            ("heads", first.heads.to_string(), shape.heads.to_string()),
            ("dim", first.dim.to_string(), shape.dim.to_string()),
            ("ctx", first.ctx.to_string(), shape.ctx.to_string()),
            ("vocab", first.vocab.to_string(), shape.vocab.to_string()),
            (
                "legal_input",
                first.legal_input.to_string(),
                shape.legal_input.to_string(),
            ),
        ];
        if let Some((field, a, b)) = axes.into_iter().find(|(_, a, b)| a != b) {
            return Err(MergeError::Mismatch {
                field,
                first: a,
                other: b,
            });
        }
    }

    // Bands, in slot order, refusing a token that would occupy two rows.
    let mut bands = Vec::new();
    let mut groups = Vec::with_capacity(shapes.len());
    for shape in &shapes {
        for band in &shape.bands {
            if bands
                .iter()
                .any(|b: &crate::chess::corpus::ConditionBand| b.token == band.token)
            {
                return Err(MergeError::DuplicateBand {
                    token: band.token.clone(),
                });
            }
            bands.push(band.clone());
        }
        groups.push(shape.bands.len());
    }

    let device = Device::Cpu;
    let loaded: Vec<HashMap<String, Tensor>> = sources
        .iter()
        .map(|path| {
            candle_core::safetensors::load(path, &device).map_err(|e| MergeError::Io {
                path: path.display().to_string(),
                message: e.to_string(),
            })
        })
        .collect::<Result<_, _>>()?;

    let mut merged: HashMap<String, Tensor> = HashMap::with_capacity(loaded[0].len());
    for (name, first_tensor) in &loaded[0] {
        if name == COND_TABLE {
            continue;
        }
        // Sum then divide, so the mean is one traversal and the
        // arithmetic is the same for any number of sources.
        let mut acc = first_tensor.clone();
        for other in &loaded[1..] {
            let t = other.get(name).ok_or_else(|| MergeError::Tensor {
                name: name.clone(),
                message: "present in one source and not another".into(),
            })?;
            if t.dims() != first_tensor.dims() {
                return Err(MergeError::Tensor {
                    name: name.clone(),
                    message: format!("{:?} and {:?}", first_tensor.dims(), t.dims()),
                });
            }
            acc = (&acc + t).map_err(|e| MergeError::Tensor {
                name: name.clone(),
                message: e.to_string(),
            })?;
        }
        let mean = (acc / loaded.len() as f64).map_err(|e| MergeError::Tensor {
            name: name.clone(),
            message: e.to_string(),
        })?;
        merged.insert(name.clone(), mean);
    }

    // The conditioning tables, stacked in slot order.
    let tables: Vec<Tensor> = loaded
        .iter()
        .zip(sources)
        .map(|(map, path)| {
            map.get(COND_TABLE)
                .cloned()
                .ok_or_else(|| MergeError::Tensor {
                    name: COND_TABLE.to_string(),
                    message: format!("{} carries no conditioning table", path.display()),
                })
        })
        .collect::<Result<_, _>>()?;
    let stacked = Tensor::cat(&tables, 0).map_err(|e| MergeError::Tensor {
        name: COND_TABLE.to_string(),
        message: e.to_string(),
    })?;
    merged.insert(COND_TABLE.to_string(), stacked);

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| MergeError::Io {
                path: parent.display().to_string(),
                message: e.to_string(),
            })?;
        }
    }
    candle_core::safetensors::save(&merged, out).map_err(|e| MergeError::Io {
        path: out.display().to_string(),
        message: e.to_string(),
    })?;

    let mut shape = first.clone();
    shape.bands = bands;
    shape.cond_groups = groups;
    shape.save(out).map_err(|e| MergeError::Io {
        path: out.display().to_string(),
        message: e.to_string(),
    })?;
    Ok(shape)
}

/// Merge single-slot checkpoints that share a base, by adding each
/// one's **difference from the base** rather than averaging them.
///
/// `base + Σ(source − base)` on every body tensor; the conditioning
/// tables are stacked as [`merge_slots`] stacks them (the base carries
/// none — it is prefix-conditioned, which is what
/// `chess_bake`'s body init requires of it).
///
/// # Why this and not the mean
///
/// The mean assumes the sources sit in one basin, and plan 11 measured
/// them at cosine 0.02 on the body — nearly orthogonal, so the
/// midpoint is neither of them. Differences from a shared base are the
/// standard answer: each source is the base plus what its condition
/// taught it, and if those two lessons are small and about different
/// things, their sum lands near both. Whether they are is exactly what
/// plan 12 measures — this function makes the claim expressible, not
/// true.
///
/// # Errors
///
/// As [`merge_slots`], plus a base that disagrees with the sources on
/// a body tensor's shape, or one that carries a conditioning table
/// (a shared base must not decide the merged model's conditioning).
pub fn merge_task_arithmetic(
    base: &Path,
    sources: &[&Path],
    out: &Path,
) -> Result<ModelShape, MergeError> {
    if sources.len() < 2 {
        return Err(MergeError::TooFewSources {
            found: sources.len(),
        });
    }
    let device = Device::Cpu;
    let base_map = candle_core::safetensors::load(base, &device).map_err(|e| MergeError::Io {
        path: base.display().to_string(),
        message: e.to_string(),
    })?;
    if base_map.contains_key(COND_TABLE) {
        return Err(MergeError::BaseIsConditioned {
            path: base.display().to_string(),
        });
    }

    // The stacking, the band bookkeeping and every refusal about the
    // sources are `merge_slots`'s. Its body mean is then replaced,
    // tensor by tensor, by the base-plus-differences form — one code
    // path owns "what a merged checkpoint is", and this owns only how
    // the bodies combine.
    let shape = merge_slots(sources, out)?;
    let mut merged = candle_core::safetensors::load(out, &device).map_err(|e| MergeError::Io {
        path: out.display().to_string(),
        message: e.to_string(),
    })?;
    let loaded: Vec<HashMap<String, Tensor>> = sources
        .iter()
        .map(|path| {
            candle_core::safetensors::load(path, &device).map_err(|e| MergeError::Io {
                path: path.display().to_string(),
                message: e.to_string(),
            })
        })
        .collect::<Result<_, _>>()?;

    for (name, base_tensor) in &base_map {
        let Some(slot) = merged.get_mut(name) else {
            // A base tensor the merged model does not want. The shapes
            // were checked against each other by `merge_slots`; this is
            // a base carrying something extra, which is ignored for the
            // same reason a restore ignores it.
            continue;
        };
        let mut acc = base_tensor.clone();
        for source in &loaded {
            let t = source.get(name).ok_or_else(|| MergeError::Tensor {
                name: name.clone(),
                message: "present in the base and not in a source".into(),
            })?;
            if t.dims() != base_tensor.dims() {
                return Err(MergeError::Tensor {
                    name: name.clone(),
                    message: format!("base {:?} and source {:?}", base_tensor.dims(), t.dims()),
                });
            }
            let delta = (t - base_tensor).map_err(|e| MergeError::Tensor {
                name: name.clone(),
                message: e.to_string(),
            })?;
            acc = (&acc + delta).map_err(|e| MergeError::Tensor {
                name: name.clone(),
                message: e.to_string(),
            })?;
        }
        *slot = acc;
    }

    candle_core::safetensors::save(&merged, out).map_err(|e| MergeError::Io {
        path: out.display().to_string(),
        message: e.to_string(),
    })?;
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::corpus::ConditionBand;
    use candle_core::DType;
    use tempfile::TempDir;

    /// A single-slot checkpoint whose every tensor holds `fill`.
    fn write_source(dir: &Path, name: &str, tokens: &[&str], fill: f32) -> std::path::PathBuf {
        let path = dir.join(format!("{name}.safetensors"));
        let device = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "wte.weight".into(),
            Tensor::full(fill, (8, 4), &device).unwrap(),
        );
        map.insert(
            "ln_f.weight".into(),
            Tensor::full(fill, (4,), &device).unwrap(),
        );
        // One row per token, each row holding its own marker so the
        // stack order is visible in the result.
        let rows: Vec<f32> = (0..tokens.len())
            .flat_map(|i| vec![fill + i as f32 + 100.0; 4])
            .collect();
        map.insert(
            COND_TABLE.into(),
            Tensor::from_vec(rows, (tokens.len(), 4), &device).unwrap(),
        );
        candle_core::safetensors::save(&map, &path).unwrap();

        let mut shape = ModelShape::compact(
            8,
            tokens
                .iter()
                .map(|t| ConditionBand::rating(0, 0, *t))
                .collect(),
        );
        shape.dim = 4;
        shape.ctx = 8;
        shape.encoding = CondEncoding::EveryPosition;
        shape.save(&path).unwrap();
        path
    }

    /// The body is averaged, the tables are stacked in source order,
    /// and the sidecar describes the result as a multi-slot model.
    #[test]
    fn a_merge_averages_the_body_and_stacks_the_tables() {
        let tmp = TempDir::new().unwrap();
        let a = write_source(tmp.path(), "a", &["<eco:B>", "<eco:C>"], 1.0);
        let b = write_source(tmp.path(), "b", &["<lo>", "<hi>"], 3.0);
        let out = tmp.path().join("merged.safetensors");
        let shape = merge_slots(&[&a, &b], &out).unwrap();

        assert_eq!(shape.cond_groups, vec![2, 2]);
        assert_eq!(shape.band_tokens(), ["<eco:B>", "<eco:C>", "<lo>", "<hi>"]);
        assert_eq!(shape.encoding, CondEncoding::EveryPosition);

        let got = candle_core::safetensors::load(&out, &Device::Cpu).unwrap();
        // Body: the mean of 1.0 and 3.0.
        let wte: Vec<f32> = got["wte.weight"].flatten_all().unwrap().to_vec1().unwrap();
        assert!(wte.iter().all(|v| (*v - 2.0).abs() < 1e-6), "{wte:?}");
        // Table: four rows, the sources' own, in order.
        let table = got[COND_TABLE].to_dtype(DType::F32).unwrap();
        assert_eq!(table.dims(), &[4, 4]);
        let rows: Vec<f32> = table.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            [rows[0], rows[4], rows[8], rows[12]],
            [101.0, 102.0, 103.0, 104.0],
            "the tables were not stacked in source order"
        );
    }

    /// Task arithmetic is base-plus-differences, and the tables still
    /// stack: with sources at base+1 and base+3, the merged body is
    /// base+4 where the mean would have been base+2.
    #[test]
    fn task_arithmetic_adds_the_differences_to_the_base() {
        let tmp = TempDir::new().unwrap();
        // The base carries a body and no conditioning table, which is
        // what a prefix-conditioned bake produces.
        let base = tmp.path().join("base.safetensors");
        let device = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "wte.weight".into(),
            Tensor::full(10.0f32, (8, 4), &device).unwrap(),
        );
        map.insert(
            "ln_f.weight".into(),
            Tensor::full(10.0f32, (4,), &device).unwrap(),
        );
        candle_core::safetensors::save(&map, &base).unwrap();
        let mut base_shape = ModelShape::compact(8, vec![ConditionBand::rating(0, 0, "<all>")]);
        base_shape.dim = 4;
        base_shape.ctx = 8;
        base_shape.save(&base).unwrap();

        let a = write_source(tmp.path(), "a", &["<eco:B>", "<eco:C>"], 11.0);
        let b = write_source(tmp.path(), "b", &["<lo>", "<hi>"], 13.0);
        let out = tmp.path().join("ta.safetensors");
        let shape = merge_task_arithmetic(&base, &[&a, &b], &out).unwrap();
        assert_eq!(shape.cond_groups, vec![2, 2]);

        let got = candle_core::safetensors::load(&out, &Device::Cpu).unwrap();
        let wte: Vec<f32> = got["wte.weight"].flatten_all().unwrap().to_vec1().unwrap();
        // 10 + (11-10) + (13-10) = 14, where the mean would be 12.
        assert!(
            wte.iter().all(|v| (*v - 14.0).abs() < 1e-5),
            "{:?}",
            &wte[..4]
        );
        // And the tables still stack rather than being differenced.
        assert_eq!(got[COND_TABLE].dims(), &[4, 4]);
    }

    /// A base with a conditioning table would decide the merged
    /// model's conditioning, and is refused.
    #[test]
    fn task_arithmetic_refuses_a_conditioned_base() {
        let tmp = TempDir::new().unwrap();
        let base = write_source(tmp.path(), "base", &["<x>"], 10.0);
        let a = write_source(tmp.path(), "a", &["<eco:B>", "<eco:C>"], 11.0);
        let b = write_source(tmp.path(), "b", &["<lo>", "<hi>"], 13.0);
        let out = tmp.path().join("ta.safetensors");
        assert!(matches!(
            merge_task_arithmetic(&base, &[&a, &b], &out),
            Err(MergeError::BaseIsConditioned { .. })
        ));
    }

    /// The disagreements a merge cannot reconcile, by name.
    #[test]
    fn a_merge_refuses_what_it_cannot_reconcile() {
        let tmp = TempDir::new().unwrap();
        let a = write_source(tmp.path(), "a", &["<eco:B>"], 1.0);
        let out = tmp.path().join("merged.safetensors");

        // One source is a copy, not a merge.
        assert!(matches!(
            merge_slots(&[&a], &out),
            Err(MergeError::TooFewSources { found: 1 })
        ));

        // The same token twice would be two rows meaning one thing.
        let same = write_source(tmp.path(), "same", &["<eco:B>"], 2.0);
        assert!(matches!(
            merge_slots(&[&a, &same], &out),
            Err(MergeError::DuplicateBand { .. })
        ));

        // A source of another width has nothing to average against.
        let wide = tmp.path().join("wide.safetensors");
        let device = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "wte.weight".into(),
            Tensor::full(1.0f32, (8, 6), &device).unwrap(),
        );
        map.insert(
            COND_TABLE.into(),
            Tensor::full(1.0f32, (1, 6), &device).unwrap(),
        );
        candle_core::safetensors::save(&map, &wide).unwrap();
        let mut shape = ModelShape::compact(8, vec![ConditionBand::rating(0, 0, "<other>")]);
        shape.dim = 6;
        shape.ctx = 8;
        shape.encoding = CondEncoding::EveryPosition;
        shape.save(&wide).unwrap();
        assert!(matches!(
            merge_slots(&[&a, &wide], &out),
            Err(MergeError::Mismatch { field: "dim", .. })
        ));

        // And a source that already carries slots is not a slot.
        let merged_once = tmp.path().join("twice.safetensors");
        let b = write_source(tmp.path(), "b", &["<lo>", "<hi>"], 3.0);
        let a2 = write_source(tmp.path(), "a2", &["<eco:B>", "<eco:C>"], 1.0);
        merge_slots(&[&a2, &b], &merged_once).unwrap();
        let c = write_source(tmp.path(), "c", &["<x>"], 5.0);
        assert!(matches!(
            merge_slots(&[&merged_once, &c], &out),
            Err(MergeError::NotSingleSlot { groups: 2, .. })
        ));
    }
}
