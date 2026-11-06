# The project in ten sentences

Internal summary for a newcomer, not preprint text. The team writes the final text in its own words.

1. Brand24 estimates the reach of a post as a constant times the author's followers; the project asks whether the
   **views** of a post can be estimated better, and with what uncertainty.
2. TikTok and YouTube publish views (their reach is the view count), X publishes them for about 75% of posts, and
   Facebook and Instagram do not publish them at all.
3. The sample is 12,500 Brand24 mentions, 348 brands, 85% English and 85% from April-August 2026, so the results describe
   this sample rather than all social media.
4. The reconstructed rule is exact on TikTok/YouTube, about **x9.3 too high on X** and **x4.1 too high on Facebook**
   (87% / 82% of posts off by more than 2x).
5. A CatBoost model on followers, likes, comments, shares and missing flags reaches MAE **0.273 / 0.456 / 0.454**
   (TikTok / YouTube / X) on log10 views - a typical error of x1.9 / x2.9 / x2.9.
6. CQR intervals restore honest coverage: the 90% band covers **0.889 / 0.902 / 0.899**, and a 50% "likely" band is
   x2.8 / x5.5 / x7.6 wide.
7. Facebook was measured on **130 hand-checked and 402 fetched posts**: the transfer has MAE **0.345 / 0.385** against
   0.93 for the rule, and the model wins in **every follower tier** (0.36-0.46 vs ~1.0).
8. What does not help: text, extra columns, graph features (including metapath2vec), multi-task and seed averaging; the
   signal is in reactions, and the author's past views when they exist.
9. Reach is **tail-dominated** (the top 1% of posts hold 60-74% of views), so brand totals need intervals and many posts,
   and alerts should use raw likes rather than the model.
10. Instagram is unmeasured, the Facebook band above 1M followers is uncalibrated, and the estimator is for levels and
    ranges - not for alerts, billing or causal claims.

Where the numbers come from: `docs/en/evidence_map.md`; the method: `docs/en/model_card.md`.
