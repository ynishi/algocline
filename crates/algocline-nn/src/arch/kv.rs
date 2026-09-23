//! Keys and values a session has already computed.
//!
//! A decode loop over a trainable architecture used to re-forward its
//! whole history at every step: generating `n` tokens ran the model
//! `1 + 2 + … + n` positions instead of `n`, because each step recomputed
//! every earlier position's keys and values from weights that had not
//! changed. The arithmetic is quadratic in the length and the work is
//! redundant — attention at position `p` reads the same `K`/`V` at every
//! earlier position it read the step before.
//!
//! A cache holds those tensors per layer and grows by the positions each
//! step adds. The model then forwards only the new tokens and attends
//! over the whole history, which is the same computation the
//! full-sequence pass performs at the last row and the standard decode
//! arrangement everywhere it appears.
//!
//! # What is cached, and what is not
//!
//! The entries are the **post-rotation, pre-broadcast** `K` and `V`:
//! `[batch, kv_heads, position, head_dim]`. Post-rotation because RoPE
//! is a function of the absolute position, which does not change as the
//! sequence grows — rotating once is correct and rotating again would
//! not be. Pre-broadcast because grouped-query attention repeats each
//! KV head across its query group, and storing the repeats would hold
//! `heads / kv_heads` copies of every tensor for no gain.
//!
//! Queries are not cached: a query belongs to the position being
//! answered and is never read again.
//!
//! # A cache belongs to one sequence
//!
//! Entries are positions of one particular sequence, so a cache carries
//! the batch it was filled at and refuses a forward at another
//! ([`KvError::BatchChanged`]). Two generations sharing a cache would
//! each attend over the other's history with every shape still
//! agreeing, which is the failure this refusal exists for — and the
//! same reason [`crate::arch::adapter::LlamaAdapter`] hands out a cache
//! per session rather than holding one.

use candle_core::{Result as CandleResult, Tensor};
use thiserror::Error;

/// A cache used in a way it cannot answer for.
#[derive(Debug, Error)]
pub enum KvError {
    /// A forward arrived at a different batch size than the one the
    /// cache holds positions for.
    #[error(
        "kv cache holds a batch of {held} and this forward is a batch of {given}; a cache \
         belongs to one sequence"
    )]
    BatchChanged {
        /// Batch the cache was filled at.
        held: usize,
        /// Batch this forward arrived with.
        given: usize,
    },
    /// A forward asked for more layers than the cache was built for.
    #[error("kv cache was built for {built} layer(s) and layer {asked} was asked for")]
    LayerOutOfRange {
        /// Layers the cache holds.
        built: usize,
        /// Layer index the model asked for.
        asked: usize,
    },
}

impl From<KvError> for candle_core::Error {
    fn from(e: KvError) -> Self {
        candle_core::Error::Msg(e.to_string())
    }
}

/// Per-layer keys and values for the positions already forwarded.
///
/// Build one with the model's own `new_cache`, which sizes it to that
/// model's layer count, and hand it to the model's `*_with_cache`
/// entry point on every step.
#[derive(Debug)]
pub struct KvCache {
    /// `[layer] -> (k, v)`, each `[batch, kv_heads, len, head_dim]`.
    /// `None` until the layer's first forward.
    layers: Vec<Option<(Tensor, Tensor)>>,
    /// Positions held. Read before a forward as the offset the new
    /// tokens start at, and advanced once after all layers have run.
    len: usize,
    /// Batch the entries were filled at, or `None` while empty.
    batch: Option<usize>,
}

impl KvCache {
    /// An empty cache for a model of `layers` layers.
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| None).collect(),
            len: 0,
            batch: None,
        }
    }

    /// Positions held — the absolute index the next forward's first
    /// token sits at.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been forwarded yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Layers this cache was built for.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Drop every entry, leaving the cache as built.
    ///
    /// For a caller reusing one cache across generations. The tensors
    /// are dropped rather than kept and overwritten: their length is a
    /// property of the sequence that filled them, so there is nothing
    /// to reuse.
    pub fn reset(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
        self.len = 0;
        self.batch = None;
    }

    /// Append this step's `k` / `v` for one layer and return the full
    /// history the attention should read.
    ///
    /// Both are `[batch, kv_heads, t, head_dim]`. The return is the
    /// concatenation along the position axis, which for the first
    /// forward of a layer is the input unchanged.
    ///
    /// Called once per layer per forward, by the model. The caller then
    /// calls [`Self::advance`] once — the length belongs to the
    /// forward, not to each layer of it.
    pub fn push(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let built = self.layers.len();
        let slot = self.layers.get_mut(layer).ok_or(KvError::LayerOutOfRange {
            built,
            asked: layer,
        })?;
        let merged = match slot.take() {
            Some((pk, pv)) => (
                Tensor::cat(&[&pk, k], 2)?.contiguous()?,
                Tensor::cat(&[&pv, v], 2)?.contiguous()?,
            ),
            None => (k.contiguous()?, v.contiguous()?),
        };
        *slot = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }

    /// Record that a forward of `t` positions at batch `batch` has
    /// completed.
    ///
    /// Separate from [`Self::push`] because the length is one fact
    /// about the forward and `push` runs once per layer; advancing
    /// inside it would count the same positions once per layer.
    pub fn advance(&mut self, batch: usize, t: usize) {
        self.batch = Some(batch);
        self.len += t;
    }

    /// Refuse a forward whose batch is not the one the entries hold.
    ///
    /// Checked before any layer is touched, so a refusal leaves the
    /// cache exactly as it was rather than half-extended.
    pub fn check_batch(&self, batch: usize) -> Result<(), KvError> {
        match self.batch {
            Some(held) if held != batch => Err(KvError::BatchChanged { held, given: batch }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn kv(batch: usize, heads: usize, t: usize, dim: usize) -> Tensor {
        Tensor::ones((batch, heads, t, dim), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn the_first_push_returns_what_it_was_given() {
        let mut cache = KvCache::new(2);
        let (k, v) = cache.push(0, &kv(1, 2, 3, 4), &kv(1, 2, 3, 4)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 3, 4]);
        assert_eq!(v.dims(), &[1, 2, 3, 4]);
        assert_eq!(
            cache.len(),
            0,
            "length advances once per forward, not per layer"
        );
    }

    #[test]
    fn a_later_push_returns_the_whole_history() {
        let mut cache = KvCache::new(1);
        cache.push(0, &kv(1, 2, 3, 4), &kv(1, 2, 3, 4)).unwrap();
        cache.advance(1, 3);
        let (k, _) = cache.push(0, &kv(1, 2, 1, 4), &kv(1, 2, 1, 4)).unwrap();
        assert_eq!(k.dims(), &[1, 2, 4, 4], "3 held + 1 new");
        cache.advance(1, 1);
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn layers_do_not_share_entries() {
        let mut cache = KvCache::new(2);
        cache.push(0, &kv(1, 2, 3, 4), &kv(1, 2, 3, 4)).unwrap();
        let (k, _) = cache.push(1, &kv(1, 2, 3, 4), &kv(1, 2, 3, 4)).unwrap();
        assert_eq!(
            k.dims(),
            &[1, 2, 3, 4],
            "layer 1 must not read what layer 0 pushed"
        );
    }

    #[test]
    fn reset_returns_the_cache_to_its_built_state() {
        let mut cache = KvCache::new(2);
        cache.push(0, &kv(1, 2, 3, 4), &kv(1, 2, 3, 4)).unwrap();
        cache.advance(1, 3);
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.layer_count(), 2);
        let (k, _) = cache.push(0, &kv(1, 2, 2, 4), &kv(1, 2, 2, 4)).unwrap();
        assert_eq!(
            k.dims(),
            &[1, 2, 2, 4],
            "nothing of the old sequence is left"
        );
    }

    #[test]
    fn a_cache_refuses_a_forward_at_another_batch() {
        let mut cache = KvCache::new(1);
        cache.advance(1, 3);
        assert!(cache.check_batch(1).is_ok());
        let err = cache.check_batch(2).unwrap_err();
        assert!(matches!(err, KvError::BatchChanged { held: 1, given: 2 }));
    }

    #[test]
    fn a_layer_past_the_end_is_refused_rather_than_silently_dropped() {
        let mut cache = KvCache::new(2);
        let err = cache
            .push(2, &kv(1, 2, 1, 4), &kv(1, 2, 1, 4))
            .unwrap_err()
            .to_string();
        assert!(err.contains("layer 2"), "{err}");
    }
}
