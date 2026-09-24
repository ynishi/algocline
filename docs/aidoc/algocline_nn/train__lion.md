# algocline-nn::train::lion

Lion — the sign-momentum optimizer from *Symbolic Discovery of
Optimization Algorithms* (Chen et al., arXiv:2302.06675).

# The update

Writing `interp(x, y, a) = (1 - a)·x + a·y`, one step is

```text
update   = sign(interp(g, m, β₁))
w        = w - lr·update - lr·λ·w
m        = interp(g, m, β₂)
```

Two properties follow from the `sign`, and both matter to a caller
coming from AdamW:

- **Every element moves by exactly `lr`.** There is no per-parameter
  scaling, so the learning rate is the step size rather than an
  upper bound on it. The paper's guidance is a learning rate 3–10×
  *smaller* than AdamW's and a decoupled weight decay 3–10× *larger*,
  so that the effective decay `lr·λ` lands in the same place.
- **One state tensor instead of two.** Lion keeps a momentum and no
  second moment, which is the memory argument for it.

The momentum update reads the *pre-step* momentum, the same value
the update direction was computed from — not the one the step just
produced. Both lines above use `m` on the right-hand side, and
swapping their order silently changes the algorithm into something
with no published behaviour.

# Precision

An FP32 master copy is kept for parameters that are not already F32.
With a sign update every element moves by exactly `lr`, so a BF16
parameter whose ulp exceeds `lr` would round every step away and
stall at its initial value while the loss curve looked merely flat.
F32 parameters are updated in place — the master would be a copy of
the thing it mirrors.

## Types

- `Lion` — Lion over candle `Var`s.
- `ParamsLion` — Lion's hyperparameters.

