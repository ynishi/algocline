//! Driving a model one block at a time.
//!
//! The ordinary forward hands back logits and keeps, inside the
//! autograd graph, every intermediate that produced them — for a
//! transformer that is on the order of ten tensors per block, all of
//! them `[batch, seq, ·]`, held until the backward pass reads them.
//! Context length and batch size are bounded by that, not by the
//! parameters.
//!
//! Gradient checkpointing trades it back: keep only each block's
//! **input**, and recompute the block's insides during the backward
//! pass, one block at a time. Memory falls from every intermediate of
//! every block to one tensor per block plus the intermediates of
//! whichever block is currently being recomputed; the cost is a second
//! forward pass, so roughly a third more compute for a fraction of the
//! activation memory. The technique is Chen et al. 2016,
//! [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).
//!
//! This trait is what lets the training loop do that: a model that
//! implements it can be run embed → block → block → … → head under the
//! loop's control instead of in one call.
//!
//! # Why the loop and not the model
//!
//! Recomputation has to be interleaved with the backward pass, and the
//! backward pass belongs to the training loop. A model that tried to
//! own both would have to own the loss as well.
//!
//! # What cannot be checkpointed here
//!
//! [`Checkpointable::checkpointable`] answers for each model. The two
//! refusals today are both about a forward that carries more than its
//! output:
//!
//! - **A mixture-of-experts block** returns a load-balancing term
//!   alongside its activations, and that term is part of the loss. A
//!   blockwise driver that dropped it would train without the balancing
//!   objective while still reporting a loss — and the routing would
//!   collapse onto a few experts with nothing saying so.
//! - **An input channel** (a conditioning table, an allowed-id set) is
//!   read at the embedding and again at every position; the blockwise
//!   surface here takes ids alone. Refused rather than run with the
//!   channel dropped, which is the same silent failure the forward path
//!   already refuses.

use candle_core::{Result as CandleResult, Tensor};

/// A model the training loop can drive one block at a time.
///
/// The three methods compose to exactly the ordinary forward:
/// `head_forward(block_forward(n-1, … block_forward(0, embed_input(xs))))`
/// must equal `forward(xs)`. That equality is what makes a checkpointed
/// run the same run — it is asserted per architecture rather than
/// assumed.
/// # Every method refuses by default
///
/// A model that has no blockwise view writes `impl Checkpointable for
/// MyModel {}` and is done: checkpointing is then refused for it, which
/// is the honest answer. The alternative — no defaults — would make the
/// trait a tax on every model that reaches the full-fine-tune entry
/// point, including the ones that will never want this.
///
/// The defaults are arranged so both ways of implementing it partially
/// fail safely. Forget [`Checkpointable::checkpointable`] and
/// checkpointing is refused rather than run against a half-built view;
/// forget one of the three forward halves and the first step fails
/// loudly rather than training something else.
pub trait Checkpointable {
    /// Blocks the loop will walk.
    fn block_count(&self) -> usize {
        0
    }

    /// Token ids to the hidden state the first block reads — the
    /// embedding, and whatever the architecture adds beside it.
    fn embed_input(&self, _xs: &Tensor) -> CandleResult<Tensor> {
        Err(no_blockwise_view())
    }

    /// One block, `[batch, seq, dim]` in and out.
    fn block_forward(&self, _index: usize, _h: &Tensor) -> CandleResult<Tensor> {
        Err(no_blockwise_view())
    }

    /// Final norm and language-model head: the hidden state to
    /// `[batch, seq, vocab]` logits.
    fn head_forward(&self, _h: &Tensor) -> CandleResult<Tensor> {
        Err(no_blockwise_view())
    }

    /// Whether this particular model can be driven this way.
    ///
    /// `Err` carries what stands in the way, phrased for the caller who
    /// asked for checkpointing — the alternative to refusing is a run
    /// that trains a different objective and reports the same loss.
    fn checkpointable(&self) -> Result<(), String> {
        Err(NO_BLOCKWISE_VIEW.into())
    }
}

/// What a model that does not decompose says when asked to.
const NO_BLOCKWISE_VIEW: &str =
    "this model does not implement a blockwise view, so its activations cannot be recomputed \
     one block at a time";

fn no_blockwise_view() -> candle_core::Error {
    candle_core::Error::Msg(NO_BLOCKWISE_VIEW.into())
}
