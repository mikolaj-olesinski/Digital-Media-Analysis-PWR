# Slide plan (M3 / client meeting)

Internal plan: which result goes on which slide, with the message and the caveat. Not preprint text; the team writes
the actual wording.

| # | slide | figure / table | message | caveat |
|---|---|---|---|---|
| 1 | Title | — | Can we estimate visibility better than a constant, with uncertainty? | — |
| 2 | Data | small table: views available (TT 100%, YT 99%, X 75%, FB 0%, IG 0%) | ground truth exists on three platforms; FB measured separately | the sample is not random (85% Apr-Aug 2026) |
| 3 | Model vs rule | `figures/cqr_bands_*` + MAE table (0.273 / 0.456 / 0.454; FB 0.385) | the model halves the error and gives honest bands | bands are wide (x14-x98) |
| 4 | FB rule vs model | `figures/fb_rule_vs_model_*` | the model wins in every follower tier (0.36-0.46 vs ~1.0) | FB only video/Reel |
| 5 | Why it is hard | `figures/reach_concentration_*` | reach is tail-dominated (top 1% = 60-74%) | the point estimate is a median, not a guarantee |
| 6 | FB intervals | `figures/fb_coverage_*` | pages below 1M are covered 0.94; above 1M 0.80-0.86 (0.86 in the figure) | do not claim 90% above 1M |
| 7 | Calibration budget | `figures/calibration_curve_*` | X needs 25-50 posts (0.650 -> 0.551); TT/YT/FB zero-shot | calibration hurts at small n |
| 8 | What to report / not | table from the client Q&A | range, never a bare point; no alerts, no billing | — |
| 9 | Limits | status "known risks" | honest scope | — |


## If you have one figure

Use **`figures/fb_rule_vs_model_*`**. It puts the model against Brand24's current rule tier by tier, on measured posts,
and the message survives without a table: the model wins everywhere (0.36-0.46 vs ~1.0). It answers the client's actual
question ("is it better than what we do?"), not an abstract MAE.

Backup: `figures/cqr_bands_*` if the question is about uncertainty.

## If you have 60 seconds

Open with the 60-second version from `client_qa.md`, show `fb_rule_vs_model`, and stop. Do not walk through the model
card.

## What not to put on a slide

- "We estimate reach" — it is views.
- A calibrated band for FB above 1M.
- The model as an alert engine.
- Instagram numbers.
- Any causal arrow.

## Figures to generate before the meeting

All are already built by `python src/result_figures.py en|pl`:
`cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`, `fb_rule_vs_model`.

## Presenting uncertainty on a slide

- Never show a bare point: draw the **50% band as a thick bar** and the **90% band as a thin line** around it (or error
  bars), and label them once in the legend.
- For a general audience, translate log10 widths into factors: "x14", "x51", "x98", not "1.14 log10".
- One sentence per number: "Estimated 12k views (likely 3k-34k, 90%: 0.9k-170k), read 2 days after posting."
- On the "model vs rule" slide, show both points and the rule's error as a reference; the model's band should visibly
  overlap the rule only rarely.
- On the FB slide, separate below/above 1M: the first has a band, the second is point-only; do not draw a band for >1M.
- Add the caveat under the figure, not in a footnote: "bands are 90% unless stated; views, not people".
