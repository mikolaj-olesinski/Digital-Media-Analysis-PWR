# Client Q&A: what to say, what not to say

Short answers for a meeting with Brand24, each backed by an experiment. Not preprint text.

## What to say

**"Is it better than our constant?"**
Yes on X and Facebook. On X the model's MAE is 0.454 (x2.8) against about 1.0 for the rule; on Facebook 0.385 against
0.93, and it wins in every follower tier (E165). On TikTok and YouTube your reach already **is** the view count, so
there is nothing to improve there.

**"Can you estimate Instagram?"**
No, not from public data. An anonymous check returned no view count for 0 of 160 posts; Instagram needs a login. For
Instagram we can report reactions, not visibility.

**"How accurate is it?"**
MAE 0.27 / 0.46 / 0.45 (TikTok / YouTube / X) — a typical factor of x1.9 / x2.9 / x2.8. On Facebook 0.385 (x2.4). The
90% band is x14 / x51 / x98; a 50% "likely" band is x2.8 / x5.5 / x7.6 (E118/E137/E141).

**"Can we use it for alerts?"**
Not for alerts. At the top 1% a ranking by raw likes (or your rule on TikTok/YouTube) beats the MAE model; the model
is for levels and intervals, not for finding the next viral post (E104/E106).

**"What about brand totals?"**
Report them with an interval and the number of posts. 5 posts give a x53 p90/p10 spread, 20 posts still x6.4 (E112).
Facebook totals are x2 low before calibration, and the 80% brand interval covers 0.74 (E157/E158).

**"How many posts must we collect to calibrate?"**
Only X needs data: 25-50 posts bring MAE from 0.650 to 0.551-0.563. TikTok, YouTube and Facebook are already calibrated
zero-shot (E115). The client does not have to collect posts for those.

**"What if our data is incomplete?"**
Retrain without the missing feature rather than feeding the full model with gaps: TikTok without reactions 1.139 ->
0.583, YouTube without followers 0.711 -> 0.494 (E163). Switch at ~35% missing reactions (TikTok/YouTube) or ~20%
missing followers (YouTube/X) (E164).

**"Why is X harder?"**
25% of X view counts are missing and not at random (older and smaller accounts), and zero reactions are common; the
model needs a wide band there (E105/E117).

**"Large Facebook pages?"**
Give the point estimate only. Above 1M followers the 90% band covers 0.80-0.86 and cannot be calibrated with our 70 posts
(E147/E151); the model under-estimates the mean by about x2.


## Answering "how accurate is it?" without a table

- "Typically within **x2-3**: x1.9 on TikTok, x2.9 on YouTube and X."
- "Nine times out of ten within **x14** on TikTok, **x51** on YouTube, **x98** on X."
- "On Facebook: x2.4 typical, and the 90% range is x98 for pages below 1M followers; above 1M we give the point only."
- "Before reactions appear: much wider, x260-900 - too early for a point."
- Do **not** say "+-20%" or "+-X views": the error is multiplicative, not additive.

## If the client asks for a screenshot of a profile or post

Do not send screenshots that show an author's name, handle, avatar or URL: the data statement commits us to aggregates
only. Offer instead: the post ID and the aggregate numbers, a crop without the author block, or a table row. If the
client needs to verify a specific post, they have the Brand24 record; we can confirm the counters we read without
reproducing personal data. A removal request (a post or an author that should not appear) is actioned immediately:
remove or aggregate, then say what changed.

## What not to say

- "It estimates **reach**" — it estimates **views**; unique reach is not public.
- "The 90% band is calibrated on Facebook above 1M" — it is not.
- "It beats likes for alerts" — it does not.
- "Instagram is validated" — it is not.
- "Brand totals are reliable from a few posts" — they are not.
- "Views cause X" — everything is correlational.
- "Use it for billing" — no; it is an estimate with a wide band.

## What we still need from the client

- Confirm the constants behind `b24_estimated_reach` (reconstructed, not confirmed).
- Answer whether a zero view count on YouTube can mean "not collected".
- Decide on Instagram: manual sample, login-based access, or a provider.


## The 60-second version

Show `figures/fb_rule_vs_model_en.png`: the model against the client's rule, tier by tier.

"We tested whether the views of a post can be estimated better than the constant times followers. On X and Facebook the
model halves the error - MAE 0.45 and 0.39 against about 1.0 - and gives an honest 90% range; on TikTok and YouTube your
reach already is the view count, so there is nothing to fix there. Use it for levels and ranges, not for alerts, and not
on Instagram, where views are not public."

## When the client asks for a number we do not have

- **"What are the views of this Instagram post?"** We do not have them. Offer reactions, or a manual sample of 100-150
  posts if the client can provide access; do not invent a number.
- **"What are the views of this ordinary Facebook post (not a video/Reel)?"** Facebook does not expose them; there is
  nothing to estimate from. Say so.
- **"How many people saw it?"** Only views are public; unique reach is not. Give views and say they are not people.
- **"Give us the brand total from these three posts."** Give the calibrated total with a very wide interval and the
  sample size, or refuse to compare; 5 posts give a x53 spread (E112).
- **"Give us the 90% band for this page above 1M followers."** Give the point only; the band is uncalibrated there
  (E147/E151). Offer a conservative range instead of a false 90%.
- **"Will this post reach 100k?"** We can give a probability-style range, not a guarantee; the tail is dominated by a
  few posts (E96).
- **"Why is your number lower than Brand24's?"** On Facebook the rule is x4.1 too high (E111); on X x9.3 (E99). We
  measured views; the rule is an estimate.
- **"Can you guarantee the estimate?"** No. It is an estimate with a wide band, not a billing figure.
