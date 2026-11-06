# Decisions

Methodological choices every experiment follows unless a decision here is changed by a person. Newest last.

| Date | Decision | Why | Changed by |
|---|---|---|---|
| 2026-10 | Target is log10(view_count) on TikTok, YouTube, X; FB/IG only through transfer | Only these have views | team |
| 2026-10 | 5-fold CV grouped by brand; brand bootstrap CI (`src/validation.py`) | A brand never in train and test; posts of a brand are not independent | team |
| 2026-10 | Metric MAE on log10 (0.3 ≈ ×2, 0.5 ≈ ×3, 1.0 ×10); report typical factor 10^MAE | Views span 7 orders of magnitude | team |
| 2026-11-02 | Features built from views use the training fold only | Avoid leakage | team |
| 2026-11-02 | Default final in-platform model: CatBoost with MAE loss, followers + reactions + flags | E11, E32 | team |
| 2026-11-04 | History and `current_*` timestamps read as Europe/Warsaw and converted to UTC | Same check is 2 h later than `metrics_updated_at` in summer (E38). To confirm with Brand24 | agent (homelab3) |
| 2026-11-04 | Forecast pairs: one (first check, last check) pair per post, gap ≥ 1 h | Equal weight per post | agent (homelab3) |
| 2026-11-04 | At posting time the horizon (hours to the check) is an allowed input | The user chooses when they want the estimate | agent (homelab3) |
| 2026-11-04 | Runs on homelab3 go through Kueue (`amc` queue) | Machine shared with two other projects | user |
| 2026-11-06 | Facebook target = `video_view_count` (the view count shown by the player), not `play_count` (plays on the Reel card) | The two counters differ (median ×3.6); views transfer without calibration (MAE 0.345 vs 0.686) and match the X/YouTube "views" construct. `plays` stays in `data/processed/fb_fetched.csv` for audit | user (Mikołaj) |
| 2026-11-06 | Report CQR intervals (50% and 90%), not the raw quantile band | Raw 5-95 quantiles cover 74%; CQR reaches 0.889-0.902 (E118, E137) | agent (deepseek) |
| 2026-11-06 | Instagram: no visibility estimate without a login or a provider; report reactions only | 0/160 anonymous checks return views; any IG number would be unvalidated | agent (deepseek) |
| 2026-11-06 | Alerts use raw likes/the rule, not the MAE model | Likes/Brand24 rank better at the top 1% (E104, E106) | agent (deepseek) |
| 2026-11-06 | If a feature is systematically missing, retrain without it (followers-only / reactions-only) | Specialised model beats the full one under missingness; switch points in E163/E164 | agent (deepseek) |
| 2026-11-06 | Facebook above 1M followers: report the point only, no calibrated 90% band | Coverage stays 0.84 with 70 posts; widening, shift, tiers and a large-page model do not fix it (E147, E151) | agent (deepseek) |
| 2026-11-06 | The FB hybrid (global <1M + large-page model >1M) is optional, not a claimed improvement | 0.385 -> 0.379, CI includes zero (E155) | agent (deepseek) |
| 2026-11-06 | Brand totals: one-number calibration + interval + report the number of posts | Totals are x2 low before calibration and unstable on few posts (E109, E112, E157, E158) | agent (deepseek) |
