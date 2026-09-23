# algocline-nn::arch::moe

Dense Mixture-of-Experts feed-forward for the GPT-2 stack.

Replaces a block's MLP with a router + `n_experts` GPT-2-shaped
experts (`c_fc → GELU → c_proj`). Routing follows the standard
top-k recipe (Switch Transformer, Fedus et al. 2021 for top-1;
Mixtral, Jiang et al. 2024 for the top-2 convention): softmax the
router logits, keep the top-k probabilities per token, renormalize,
and mix expert outputs with those weights.

**Dense compute only.** Every expert runs on every token and the
outputs are combined by weight — no token dispatch / grouped GEMM.
candle has no expert-dispatch kernel, and a scatter/gather custom op
would sit in the same no-backward `CustomOp` trap the LayerNorm /
RoPE slow-path shims exist for (`apply_slow_layer_norm`,
`arch::tinyllama`). The dense mixture is a composition of
softmax / linear / GELU / mul / add, all of which carry proper
backward implementations, so the autograd chain stays intact.

`top_k = n_experts` degenerates into a dense softmax mixture where
every expert receives gradient on every step — the
`tests/moe_grad_coverage.rs` gate runs in that mode to check
structural gradient reachability without tangling with the (normal)
sparsity of top-k routing.

## Types

- `MoeConfig` — Configuration for the dense-MoE feed-forward.

