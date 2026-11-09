# Repository description (for GitHub "About")

Copy-paste text; English only, because the GitHub description is a single English field.

**Description (one sentence):**

> Estimating the visibility (views) of social media posts from followers and reactions, better than a constant and with
> calibrated uncertainty — an AMC course project for Brand24.

**Topics:**

`social-media` `reach-estimation` `conformal-prediction` `catboost` `brand24` `amc-pwr`

**Highlights (three bullets):**

- The model beats the constant rule: MAE 0.27 / 0.46 / 0.45 (TikTok / YouTube / X) on log10 views against about 1.0; on
  402 measured Facebook posts 0.385 against 0.93.
- CQR intervals are calibrated in-platform (90% coverage 0.89-0.90), with a 50% "likely" band and a 90% band.
- Honest scope: Instagram is unmeasured, the Facebook band above 1M followers is uncalibrated, and alerts should use raw
  likes rather than this model.

Where to start: `README.md` -> `docs/en/story.md` -> `docs/en/status.md`.
