# algocline-nn::train::scheduler

Learning-rate schedules for the training loop.

Trainers ask for a scheduler once at construction time and then call
[`Scheduler::lr_at`] each step to get the current learning rate.
The loop then passes the value through to the optimizer via
`AdamW::set_learning_rate`.

Two schedules ship today: a plain constant one (useful for tests and
sanity checks) and a cosine schedule with linear warmup that matches
the nanoGPT / HF Trainer default. Both live behind the [`Schedule`]
enum so a config can pick between them via a plain string field.

## Types

- `ScheduleKind` — Which schedule variant the trainer requested.
- `Scheduler` — Learning-rate schedule state carried across steps.

