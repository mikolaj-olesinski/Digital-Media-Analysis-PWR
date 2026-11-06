# FAQ for a newcomer

Short answers to the questions that come up first, for the team. For client-facing wording see `client_qa.md`. Numbers and experiment IDs: `docs/en/evidence_map.md`.

**Why is there no Instagram result?**
Instagram does not expose a view count without a login: an anonymous check returned views for 0 of 160 posts. Likes and
comments for 115 of 160 were recorded, but without views there is nothing to validate a visibility estimate against.
The options are a manual sample, a logged-in account or a provider (question Q12).

**Why is the X band so wide (x98) if the model is fine?**
X has 25% missing view counts, not at random (66% of 2025 posts vs 22% of 2026), and 64% of posts have zero reactions
(E105). A 90% band that covers those cases has to be wide; a 50% "likely" band is x7.6 (E137).

**Why not use the model for alerts?**
At the top 1% a ranking by raw likes, or by the Brand24 rule on TikTok/YouTube, beats the MAE model: precision 0.64 vs
0.56 on TikTok, 0.64 vs 0.48 on YouTube, 0.47 vs 0.26 on X (E104); at a fixed alert budget the model needs many more
alerts (E106). The model is good at levels, not at finding the next viral post.

**Why is the Facebook band uncalibrated above 1M followers?**
Only 70 fetched posts come from pages above 1M. Coverage stays at 0.80-0.86 even after widening, a level shift, per-tier
quantiles or a dedicated large-page model (E147/E151). With that sample the tier cannot support a 90% claim.

**Why does the graph not help?**
The edges we have are co-posting edges (author-brand-hashtag), not diffusion edges. The mean residual of neighbouring
training posts does not predict a post's residual (author rho <= 0.07, brand/hashtag ~0; E128); hand features, GraphSAGE,
spectral embeddings and metapath2vec all fail or hurt (E65/E122/E127/E128).

**Why does text not help?**
Reactions already carry the signal. Text adds <= 0.01 next to reactions (E100); without reactions it helps more, but
at posting time the model is still much worse than with reactions (E43/E130).

**Why are zero reactions not treated as missing?**
On YouTube, marking zeros as missing changes MAE by 0.003 (E116); the zeros are small posts, not collection gaps. A 0/0
post at the first check still gets views: median 5 on X, 60 on TikTok (E117/E126).

**Why are brand totals so unstable?**
Reach is tail-dominated: the top 1% of posts hold 60-74% of views (E96). Five posts give a x53 p90/p10 spread of the
brand total; even 20 posts give x6.4 (E112). Calibration and intervals help, but not enough to compare brands on a few
posts.

**Why is there no causal claim?**
Everything is observational, and the sample is not random; the model describes associations in this sample. Words like
"causes" or "drives" are not supported by the design.

## Common misunderstandings

Concepts only; which number to quote and where is in `doc_status.md` ("Typical mistakes when quoting numbers").

- **"Views are reach."** Views are not unique people; the platforms do not publish unique reach for arbitrary posts.
- **"MAE 0.3 means 30% off."** It is on log10: 0.3 is a factor of x2, 0.5 is x3, 1.0 is x10.
- **"The 90% and 50% bands are the same claim."** The 50% band is the likely range; the 90% band is the honest worst case.
- **"Facebook numbers are 0.345."** That is the 130 hand posts; the full 402-post sample is 0.385 and is the number to quote.
- **"The model beats the rule everywhere."** It does on levels (MAE); on ranking at the top 1% the raw likes/rule win.
- **"It works on Instagram."** It has never been validated there; there are no public views.
- **"The >1M band is 90%."** It is uncalibrated (0.80-0.86); report the point only.
