# algocline-nn::train::scheduler

Learning-rate schedules for the training loop.

Trainers ask for a scheduler once at construction time and then call
[`Scheduler::lr_at`] each step to get the current learning rate.
The loop then passes the value through to the optimizer via
`AdamW::set_learning_rate`.

Four schedules ship: a plain constant one (useful for tests and
sanity checks), cosine with linear warmup (the nanoGPT / HF Trainer
default), linear with warmup, and warmup-stable-decay. All live
behind the [`ScheduleKind`] enum so a config can pick between them
via a plain string field.

The three warmup-bearing schedules share one warmup ramp and differ
only in what they do afterwards, which is why they sit in one enum
rather than in separate types: a caller switching between them is
changing the tail of a curve, not the kind of thing it is.

## Types

- `ScheduleKind` — Which schedule variant the trainer requested.
- `Scheduler` — Learning-rate schedule state carried across steps.

