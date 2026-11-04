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
