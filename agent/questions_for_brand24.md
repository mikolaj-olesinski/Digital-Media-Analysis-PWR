# Questions for Brand24

Only the client can answer these. Agents add questions; people send them and record the answer with the date.

| # | Question | Why it matters | Asked | Answer |
|---|---|---|---|---|
| 1 | Can zero likes and comments on YouTube mean "not collected" or "hidden"? | 23% of YouTube posts, 35% of error (E27) | | |
| 2 | Is our reconstruction of the reach rule right (k × reactions vs c(followers) × followers, k = 700 on X)? | E24 | | |
| 3 | Are `history_*_at` and `attributes_last_update` in Polish local time, and `metrics_updated_at` in UTC? | We assume so (E38) | | |
| 4 | Do `history_k` values mean the metrics seen at that check, or the previous state? Are reactions re-collected at every check? In a few TikTok rows a history check has fewer views than the main check at the same time; on X reactions changed in 1% of re-checked posts and views in 10% | Affects E39-E42, E45 | | |
| 5 | Why do only TikTok and X have history, and only from 2026-08? | Scope of the forecast results | | |
| 6 | Is forecasting reach at posting time useful? | Decides whether E40-E43 matter to the client | 2026-11-03 (email) | "Projection of reach is always welcome, the question is whether it can be done credibly" |
| 7 | On X, posts from 2025 almost never have a view count (12 of 170), and missing counts go with a late check. Are X views collected only for recent posts, or lost after a while? Are the 63 X posts with 0 views true zeros? | Decides whether missing/zero views are dropped or modelled (E48) | not yet | |
| 8 | Does Brand24 keep view counts of an author's past mentions? Could a sample with several posts per author be shared? An author's earlier posts predict how far the next post beats the reaction-based estimate (Spearman about 0.7) | Author history gains 0.10-0.16 MAE on posts that have it (E68) | not yet | |
