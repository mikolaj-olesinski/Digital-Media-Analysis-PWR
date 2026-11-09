# Verdict: can visibility be predicted, and are we better than Brand24?

Internal, honest summary of the answers the team will be asked. Not preprint text. Numbers: `evidence_map.md`.

## Can visibility be predicted?

- **Typical post of a brand:** yes, as an order of magnitude with a range. MAE 0.273 / 0.456 / 0.454 (TikTok / YouTube /
  X) = x1.9 / x2.9 / x2.8; the 90% band is x14 / x51 / x98 (E118/E141).
- **One specific post, exact number:** no. On X the 90% band spans two orders of magnitude.
- **A fresh post with no reactions:** only a range; MAE 0.60-0.76 and a x260-900 band (E130).
- **Which post goes viral:** no. Raw likes and the Brand24 rule rank better at the top 1% (E104/E106).
- **Instagram:** no measurement at all (0/160 anonymous).
- **Facebook above 1M followers:** point only, no calibrated band (E147/E151).
- **The ceiling is close:** the noise floor is 0.22 / 0.39 / 0.33 (E56); text, graph, columns and multi-task add <= 0.01.

## Are we better than Brand24?

- **Against their exported column (`b24_estimated_reach`), on X and Facebook: yes.** X MAE 0.454 against 1.08 (58% lower),
  Facebook 0.385 against 0.93 (59% lower), and the model wins in every follower tier (E165). We also give intervals, which
  the column does not.
- **On TikTok and YouTube: no, and there is nothing to win.** Their column equals the view count there (100% / 99% of
  rows), i.e. it is a measurement with zero error; our model has 0.273 / 0.456.
- **On ranking/alerts: they are better** (likes and their rule beat the model at the top 1%).
- **On Instagram: they have a number, we have none.**
- **Against their internal models: unknown.** We only see the exported rule, not what Brand24 runs internally.

## What if the author's history is unknown?

- The model stays the **base model** (followers, reactions, flags, age). The author feature helps only the 1-8% of posts
  whose author is seen (E119: X 0.530 -> 0.430, YouTube 0.465 -> 0.446).
- **Brand history does not substitute** (E169): a temporal split with a smoothed brand prior changes MAE by -0.005
  (TikTok), -0.035 (YouTube) and +0.006 (X, CI includes 0); only 35-52% of test posts have a known brand.
- Substitutes that do work: one earlier **view** check of the same post (E40: 0.165 / 0.188 / 0.269), and calibration on
  25-50 client posts on X (E115).

## One sentence

Visibility is predictable as a level and a range for a typical post, better than Brand24's rule on X and Facebook; it is
not predictable as an exact number, not for a fresh post, not for virality, and not at all on Instagram.
