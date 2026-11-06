# Preprint outline (≤ 8 pages)

Internal structure for the team. The text is written by people; this page only says what goes where and which
experiment backs it. Numbers: `docs/en/evidence_map.md`.

## 1. Introduction (~0.75 page)

- Visibility is a commercial asset; platforms hide most of it (Brand24 context).
- The current practice: `reach ≈ constant × followers`; the reconstructed rule.
- Question: can views be estimated better, and with what uncertainty, where views exist (TikTok, YouTube, X) and where
  they do not (Facebook, Instagram)?
- Contributions: (a) a validated estimator with calibrated intervals; (b) the first measured FB check (130 + 402 posts);
  (c) honest negative results (text, graph, multi-task, alerts); (d) a decision guide for the client.

## 2. Related work (~0.75 page)

- Reach/popularity prediction; engagement as a proxy; the transfer problem.
- Conformal prediction and calibration under shift.
- Graph and text methods for popularity; why they need interaction data.
- Sources: `docs/en/literature_notes.md` (checked vs search-only marked there).

## 3. Data (~1 page)

- Brand24 sample: 12,500 mentions, 348 brands, time skew (85% Apr-Aug 2026), 85% English.
- Views availability: TikTok 100%, YouTube 99%, X 75%, Facebook 0%, Instagram 0%.
- Facebook measurement: 130 hand-checked, 402 fetched; views vs plays (x3.6); 68 URLs gone.
- AI labels: three coders, kappa 0.77-0.79 / 0.71-0.73, machine-machine only.
- E93, E95, E105, E153; data statement (the sample description).

## 4. Methods (~1.25 pages)

- Target log10(views); age of the check; views vs plays.
- Features: followers, reactions, missing flags; age for X.
- Model: CatBoost MAE; why not more families (E114/E124).
- Validation: brand-grouped CV, repeated CV (E114), temporal (E46/E136), stricter grouping (E55), bootstrap.
- Intervals: CQR (E118), 50% band (E137), temporal check (E142), FB (E123).
- Transfer: leave-one-platform-out (E131), calibration budget (E115).
- Noise ceiling (E56) and why small gains are not chased.

## 5. Results (~2.5 pages)

1. **Model vs rule** — 0.273/0.456/0.454 vs ~1.0; FB 0.385 vs 0.93; every FB tier (E165). Figures `cqr_bands`,
   `fb_rule_vs_model`.
2. **Uncertainty** — 90% coverage 0.89-0.90 in-platform, 0.925 on FB <1M, 0.80-0.86 above 1M (E118/E123/E147/E148/E151).
   Figure `fb_coverage`.
3. **Transfer** — cost of no target data: widths 1.51/2.84/3.69 (E131); the X calibration budget (E115).
   Figure `calibration_curve`.
4. **What does not help** — text (E100), extra columns (E121), graph (E122/E127/E128), multi-task (E113), seed averaging
   (E134). One short subsection, not a table dump.
5. **Author history** — +0.10 on seen authors (E68/E119); the lever for Brand24's repeated authors.
6. **Concentration** — top 1% = 60-74% of views; why point estimates are unstable (E96). Figure `reach_concentration`.
7. **Missing data** — 25% X views missing (E105); zeros (E116/E117); retraining rule (E163/E164).
8. **Brand totals** — calibration + intervals; small samples (E109/E112/E157/E158).
9. **Alerts** — likes/rule beat the model at top 1% (E104/E106).

## 6. Discussion (~0.75 page)

- What Brand24 should do per platform (one-pager).
- Where the model must not be used (alerts, billing, Instagram, >1M band).
- The size of the prize: the rule is x4-x9 off, the model x2-x3.

## 7. Limitations (~0.5 page)

- Sample not random, time skew, FB stratification, IG unmeasured, machine-only labels, reconstructed constants, no
  causal claims. Full list: status "known risks".

## 8. Reproducibility (~0.25 page)

- `src/`, `data/processed/*_numbers.json`, `docs/en/repro_map.md`; seeds; two scripts re-run byte-identical.

## Figures and tables

| item | source |
|---|---|
| fig 1 CQR bands | `figures/cqr_bands_*` |
| fig 2 FB coverage | `figures/fb_coverage_*` |
| fig 3 concentration | `figures/reach_concentration_*` |
| fig 4 calibration | `figures/calibration_curve_*` |
| fig 5 rule vs model FB | `figures/fb_rule_vs_model_*` |
| tab 1 data and views | E01, data statement |
| tab 2 model vs rule per platform | E114/E93/E95/E111 |
| tab 3 what does not help | E100/E113/E121/E122/E127/E128 |
