# Project status, 2026-11-08

Internal summary for the team. Not preprint text. Full evidence: `docs/en/evidence_map.md`, `agent/findings.md`.

## What the project does

Estimates the **visibility (views) of a social post** from followers and reactions, better than a constant, and with an
interval. TikTok, YouTube and X have public views and serve as ground truth; Facebook was measured on 130 hand-checked
and 402 fetched posts; Instagram has no public views.

## Where the model stands

| platform | MAE log10 | typical error | 90% band | rule (Brand24) |
|---|---|---|---|---|
| TikTok | 0.273 | x1.9 | 1.14 (x14) | exact (is views) |
| YouTube | 0.456 | x2.9 | 1.71 (x51) | exact (is views) |
| X (v3) | 0.454 | x2.9 | 1.99 (x98) | x9.3 too high |
| Facebook (402) | 0.385 | x2.4 | 1.99 (x98) | x4.1 too high |

- **50% bands:** 0.45 / 0.74 / 0.83 (E137/E141).
- **Posting time (no reactions):** MAE 0.60-0.76, band 2.4-3.0 (x260-900); reactions are worth waiting for.
- **Validation:** brand-grouped CV, repeated CV, temporal rolling-origin, stricter author/copy grouping; all consistent.
- **Noise ceiling:** ~0.22 / 0.39 / 0.33; the model is close on TikTok and 0.06-0.14 above on YouTube/X.


**External checks (2026-11-08).** Three independent YouTube samples were fetched with `yt-dlp`: brand videos (53), a
diverse 131-video sample (popular to small), and 47 third-party mention videos with brand channels excluded. On the
mention-like population - the one the model targets - the error is close to the export (MAE 0.690 low, 0.441 mid) and
only the popular stratum fails (1.569, ~x30 low); the x5 shift on brand videos is a sample effect, not a model failure.
Zero-view videos are reachable among obscure third-party posts but not through brand channels. X, TikTok and Instagram
still need an API, a login or a manual sample (E189-E191).

## Main conclusions

1. The constant/step rule is far off on X and FB (87% / 82% of posts off by >2x); it is exact on TikTok/YouTube.
2. Reactions are the signal; followers are a large-page feature on FB and essential on YouTube/X (E149/E162).
3. Text, labels, time, country, graph methods add <= 0.01; the graph line is closed (E127/E128).
4. The author's past views are the largest lever when available (X 0.530 -> 0.430 on seen authors), but only 1-8% of
   posts have them (E68/E119).
5. Reach is tail-dominated: top 1% = 60-74% of views; 2-4% of posts give 80% (E96).
6. Alerts should use raw likes/Brand24, not the MAE model (E104/E106).
7. Brand totals need intervals and many posts (E109/E112/E157/E158).
8. Missing data: retrain without the missing feature; concrete switch points (E163/E164).
9. CQR is the interval method; the raw quantile band under-covers (74%).
10. The >1M FB tier cannot be calibrated with 70 posts; report the point, not a band (E147/E151).


## Changes to the estimator since E94

The core is unchanged: CatBoost with MAE loss on followers, likes, comments, shares and missing flags. What changed is
the surrounding contract. **X** now includes log(age) (v3): MAE 0.467 -> 0.454 and the 90% band 2.07 -> 1.99
(E139/E141). **Intervals** are CQR at 50% and 90% instead of raw quantiles, which covered only 74% (E118/E137). The
**author's shrunk mean views** is an optional feature when the history exists (E119/E124). **Facebook** moved from a
statement to a measured transfer (0.385 on 402 posts) with a calibrated band below 1M followers and a point-only rule
above 1M (E95/E123/E147). **Missing data** has explicit switch points (E163/E164), and the **confidence flag** from
metadata is part of the output (E108).

## Open questions

- Instagram visibility: needs a login or a paid provider; decision pending.
- Facebook ordinary posts (not video/Reel): no view count; cannot be validated.
- Unique reach (people, not views): not observable from public data.
- False brand matches: needs 100-200 hand labels.
- Brand24 constants: reconstructed, not confirmed by the client.

Facebook ordinary posts and unique reach cannot be closed by more analysis: they are limits of the public data. The
other three need a decision, hand labels or the client.

## Documents

- **The estimator:** `model_card.md`, `estimator_spec.md`, `reporting_guide.md`.
- **Evidence:** `evidence_map.md`, `experiments.md` (register), `agent/vault/` (one card per experiment),
  `data/processed/*_numbers.json`.
- **For the client:** `one_pager.md`, `client_qa.md`, `slides.md`.
- **For the team:** `whats_new.md`, `future_work.md`, `doc_status.md`, `preprint_outline.md`.
- **Reproduce:** `reproduce.md`, `repro_map.md`, `scripts.md`.

## Reproducibility

- 198 experiment cards (199 experiments including the M1 exploration); every `numbers:` file and notebook referenced by a card exists.
- Re-ran `src/fb_rule_vs_model.py` and `src/cqr_50.py` end to end: the output JSON files are byte-identical to the stored
  ones (fixed seeds).
- All scripts compile (`python -m py_compile src/*.py`).
- Figures are built by `src/result_figures.py` (`cqr_bands`, `fb_coverage`, `reach_concentration`, `calibration_curve`,
  `fb_rule_vs_model`).

## Key decisions

- Target: log10 views; on Facebook `video_view_count`, not plays.
- Intervals: CQR; report a 50% and a 90% band, never the raw quantile band.
- Alerts: raw likes / the rule, not the MAE model.
- Instagram: no visibility estimate without a login or a provider.
- Missing data: retrain without the missing feature.
- Facebook above 1M followers: point only.
- Brand totals: calibration + interval + the number of posts.

Full list: `agent/decisions.md`.

## Known risks for the preprint

1. The sample is not random and is time-skewed (85% April-August 2026); results describe Brand24's sample, not all social
   media.
2. The 402 FB posts are stratified by reactions and 68 URLs were gone; the 130 hand posts are slightly easier (E153).
3. Instagram is unmeasured; any IG visibility claim would be unsupported.
4. The FB >1M band is uncalibrated (E147/E151); do not claim 90% coverage there.
5. The model is not an alert engine: likes/Brand24 rank better at the top 1% (E104/E106).
6. Brand totals on few posts are unstable (E112) and their intervals under-cover on FB (E158).
7. The AI labels are validated machine-to-machine only; there is no human kappa.
8. Brand24's constants are reconstructed, not confirmed by the client.
9. Everything is correlational; do not phrase it causally.
10. The preprint text must be written by the team, with the AI use disclosed.

## Plan coverage

Every item of the "Ideas to stand out" plan is run, closed or blocked: Facebook measured, Instagram blocked, calibration,
intervals, graph (closed), smaller tests, extremes, YouTube zeros and reaction-tier intervals all done. Details:
`plan_status.md`.

## What to bring to a meeting

- `one_pager.md` (one page) and `client_qa.md` (answers and "do not say").
- Three figures: `cqr_bands`, `fb_rule_vs_model`, `fb_coverage`.
- `slides.md` for the order of the story.

## Next steps

1. Team writes the preprint/poster (agents do not write the final text).
2. Decide on Instagram (manual sample, login, or a provider).
3. If time: monthly brand totals (B25), multi-language check (B26).
4. Ask Brand24 the open questions (`questions_for_brand24.md`).
