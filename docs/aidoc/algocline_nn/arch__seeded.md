# algocline-nn::arch::seeded

Parameter initialisation a seed decides.

Two runs of the same config differ in two places: the order the rows
arrive in, and the numbers the parameters start at. The first is
[`DatasetOpts::seed`](crate::train::DatasetOpts::seed). This is the
second.

# Why not `Device::set_seed`

candle has one, and on CUDA and Metal it works. The CPU backend
answers `cannot seed the CPU rng with set_seed` and draws from the
thread RNG (`candle-core 0.11`, `cpu_backend/mod.rs`), so on the
device most development happens on there is nothing to seed. A run
whose reproducibility depended on the backend would be repeatable on
the GPU and not on the laptop, which is the worse of the two
failures: the discrepancy only shows up once the two are compared.

# What this does instead

[`seeded_var_builder`] hands the model a [`VarBuilder`] whose
backing store draws the initial values itself, from a seeded
[`StdRng`], honouring the [`Init`] hint each parameter was declared
with. The architectures are untouched: they already say what
distribution each parameter wants — `Randn { stdev: INIT_STDEV }`
for GPT-2's embeddings, Kaiming for a linear — and this reads that
declaration rather than restating it. A separate walker that re-drew
parameters after construction would be a second copy of every
architecture's initialisation, free to drift from the first.

# What is still not deterministic

Initialisation and row order, both fixed here, are what a caller
controls. They are not everything:

- **Reduction order on the GPU.** Floating-point addition is not
  associative, and CUDA kernels do not promise a fixed summation
  order between runs. Two identically-seeded runs can diverge in the
  last bits and then, through a few thousand steps, visibly.
- **cuDNN algorithm selection**, which can vary with available
  memory.
- **Dropout and any other sampling inside a forward pass**, which
  draw from the device RNG this does not reach.

This is the field's ordinary position rather than a shortfall
peculiar to this crate — PyTorch ships the same knobs and declines
the same guarantee ("completely reproducible results are not
guaranteed across PyTorch releases, individual commits, or different
platforms"). What matters is that the part a caller can control is
controllable, and that the rest is written down.

## Functions

- `seeded_var_builder` — A [`VarBuilder`] over `vm` whose fresh parameters are drawn from

