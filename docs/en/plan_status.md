# Plan coverage: what was planned and what happened

Maps the "Ideas to stand out" items (A-I) of `project_plan.md` to their outcome. Every item was run, closed or blocked on
data; none is silently open.

| item | status | evidence |
|---|---|---|
| **A. Own Facebook/Instagram sample** | **partial**: Facebook measured (130 hand + 402 fetched), Instagram blocked (0/160 anonymous, needs a login or a provider) | E93, E95, E111, E123, E153, E166; data statement |
| **B. Calibration recommendation** | **done**: X needs 25-50 posts (0.650 -> 0.551); TT/YT/FB are zero-shot | E115 |
| **C. Intervals and abstention** | **done**: CQR 50%/90%, abstention curve from the band width | E58, E118, E137 |
| **D. False brand matches** | **blocked**: needs 100-200 hand labels from a person (not the client); examples documented, no detector | E37 (legacy), backlog B15 |
| **E. Community as a bridge to FB/IG** | **closed**: the graph does not carry residual signal; co-posting edges are not diffusion edges | E69, E122, E127, E128 |
| **F. Smaller tests** | **done**: posts without reactions, virality top 10%, copy-paste campaigns | E48, E63, E64 |
| **G. Improve the extremes** | **done**: quantiles, recalibration, tail weighting, hurdle, tier shifts; nothing beats the plain model enough to keep | E57, E114, E135, E140 |
| **H. YouTube zeros** | **done**: treating them as missing changes MAE by 0.003; zeros are small posts | E116, E117 |
| **I. Interval by reaction tier** | **done**: one width per tier restores coverage; CQR is now the main method | E34, E118 |

## What this means

- **A** is the only item still open, and only for Instagram; the Facebook half is the most novel result of the project.
- **D** is the only blocked analysis that needed human labelling.
- Everything else was tested and, where it failed, the failure is recorded in the register and the register row says so.

What is open and prioritized next: `future_work.md`.
