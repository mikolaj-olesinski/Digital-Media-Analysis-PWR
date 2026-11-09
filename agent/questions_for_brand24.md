# Questions for Brand24

Only the client can answer these. Agents add questions; people send them and record the answer with the date.

| # | Question | Why it matters | Asked | Answer |
|---|---|---|---|---|
| 1 | Can zero likes and comments on YouTube mean "not collected" or "hidden"? | 27% of YouTube posts, 35% of error (E27); treating zeros as missing gains only 0.003 (E116), so the answer decides whether to trust them | | Impact measured: 0.003 MAE (E116); the semantics still need the client |
| 2 | Is our reconstruction of the reach rule right (k × reactions vs c(followers) × followers, k = 700 on X)? | E24; measured: X median x9.3 high, 87% of posts off >2x (E99), FB x4.1 (E111) | | Reconstruction measured (E99/E111); confirmation still needed |
| 3 | Are `history_*_at` and `attributes_last_update` in Polish local time, and `metrics_updated_at` in UTC? | We assume so (E38) | | |
| 4 | Do `history_k` values mean the metrics seen at that check, or the previous state? Are reactions re-collected at every check? In a few TikTok rows a history check has fewer views than the main check at the same time; on X reactions changed in 1% of re-checked posts and views in 10% | Affects E39-E42, E45 | | |
| 5 | Why do only TikTok and X have history, and only from 2026-08? | Scope of the forecast results | | |
| 6 | Is forecasting reach at posting time useful? | Decides whether E40-E43 matter to the client | 2026-11-03 (email) | "Projection of reach is always welcome, the question is whether it can be done credibly" |
| 7 | On X, posts from 2025 almost never have a view count (12 of 170), and missing counts go with a late check. Are X views collected only for recent posts, or lost after a while? Are the 63 X posts with 0 views true zeros? | Measured: 25.4% of X views missing, 66.5% of 2025 vs 22.4% of 2026 (E105); decides whether missing/zero views are dropped or modelled (E48) | not yet | Measured 25.4% missing, 66.5% of 2025 vs 22.4% of 2026 (E105); the collection policy still needs the client |
| 8 | Does Brand24 keep view counts of an author's past mentions? Could a sample with several posts per author be shared? An author's earlier posts predict how far the next post beats the reaction-based estimate (Spearman about 0.7) | Author history gains 0.10-0.16 MAE on posts that have it (E68) | not yet | |
| 9 | Is paid promotion (boosted posts, ads) known for any mentions? | Posts ×10 above the reaction-based estimate are more often promo, copies and brand accounts (E74); would confirm the paid-reach reading | not yet | |
| 10 | Do you see or store the view count of Facebook and Instagram Reels/videos internally, even when it is not in the export? If yes, could you export our 312 hand-check posts with views **and** reactions read at the same timestamp? | Replaces the blocked manual check (B14, E82); FB is now 130/312 hand-checked and 402 posts fetched publicly (E93/E95), IG still blocked | not yet | |
| 11 | On Facebook, which counter is "reach": the plays shown on the page or `video_view_count`? They differ by a median x3.6 in our sample (E95) | Decides the FB target definition (we use `video_view_count`) | not yet | |
| 12 | Can we get Instagram views at all — a manual sample, a logged-in account, or a provider? | Without it, IG visibility cannot be validated; 0/160 anonymous attempts (data statement) | not yet | |
