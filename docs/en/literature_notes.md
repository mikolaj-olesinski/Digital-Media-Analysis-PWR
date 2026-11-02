# Literature notes

Quick scan from 2026-11-02, done with AI agents, not a systematic review. Read a source in full before citing it in the preprint. Status: **checked** means title, authors, year and the stated claim were confirmed on the source page or arXiv abstract page; **search only** means seen in search results and not confirmed.

## Checked
| Source | What it says | Relation to our work |
|---|---|---|
| Shulman, Sharma, Cosley, 2016, *Predictability of Popularity: Gaps between Prediction and Understanding*, ICWSM. [arXiv:1603.09436](https://arxiv.org/abs/1603.09436) | With early peeking, models generalise across datasets (music, books, photos, URLs). Most power comes from temporal signals; for other features even the direction of the correlation is not consistent across domains | Close to ours in spirit: post-publication signal dominates and transfers. Different signal (speed of first adopters, not reactions) |
| Sahu et al., 2026, *Benchmarking Multi-Modal Graph-based Social Media Popularity Prediction*. [arXiv:2606.27539](https://arxiv.org/abs/2606.27539) | Benchmark on four Bluesky and Reddit datasets; reports cross-platform generalisation; graphs of social interactions | Possible counterpoint on graphs: they use real interaction graphs, ours is text similarity. Abstract only; the agent's finding on pooling is not confirmed |
| Tibshirani, Barber, Candès, Ramdas, 2019, *Conformal Prediction Under Covariate Shift*. [arXiv:1904.06019](https://arxiv.org/abs/1904.06019) | Weighted conformal intervals when train and test features differ, using the likelihood ratio | Method for plan item C. It addresses shift in features; our level shift is a change in the relation itself, so it may not be enough |
| Sokol, Moniz, Chawla, 2024, *Conformalized Selective Regression*. [arXiv:2402.16300](https://arxiv.org/abs/2402.16300) | The model abstains when uncertain, with confidence from conformal prediction, and an evaluation framework | Direct method for plan item C (abstention) |
| Mannocci, Mazza, Monreale, Tesconi, Cresci, 2024, *Detection and Characterization of Coordinated Online Behavior: A Survey*, ACM Computing Surveys 2026. [arXiv:2408.01257](https://arxiv.org/abs/2408.01257) | Survey of coordinated behaviour | Reference for plan item F (campaigns). Its use of text-similarity networks is not confirmed from the abstract |
| Bekavac, Mayer, 2026, *Platforms' Research API Data Access: What Users See vs. What Researchers can Retrieve*. [arXiv:2601.12390](https://arxiv.org/abs/2601.12390) | Research APIs exclude up to about 50% of what users see and strip up to about 83% of metadata, with limits down to about 1,000 requests a week (maximum figures, not split by platform) | Limitation for FB/IG validation |
| Siegel, 2019, *Introducing the Brandwatch Reach Score*, company blog. [Link](https://www.brandwatch.com/blog/introducing-the-brandwatch-reach-score-making-reporting-simpler-and-more-trustworthy/) | Reach from typical likes, retweets and shares relative to followers; other sources through regression on likes plus assumptions. No validation against real views described | Vendor, not research. Same family of idea as the Brand24 rule |
| SocialCrawl, 2026, *Instagram Reels Views: What Counts*, company blog. [Link](https://www.socialcrawl.dev/blog/how-instagram-counts-views) | Convenience sample of 36 accounts, 425 posts: 196 of 197 videos show public views, 0 of 228 images or carousels | Vendor, not research. Supports plan item A only for video, not for ordinary posts |
| Rubio-Madrigal, Burkholz, 2026, *Fixed Aggregation Features Can Rival GNNs*, ICML 2026. [arXiv:2601.19449](https://arxiv.org/abs/2601.19449) | Fixed neighbour aggregations plus a tuned MLP match or beat GNNs on 12 of 14 benchmarks | Basis for variant A in notebook 12. Benchmarks differ from our task |
| Luan et al., 2023, *When Do Graph Neural Networks Help with Node Classification?*, NeurIPS 2023. [arXiv:2304.14274](https://arxiv.org/abs/2304.14274) | GNN gains depend on node distinguishability, not only on homophily | Reminder that homophily in our text graph does not guarantee a gain. Node classification, not regression |
| Roy, Mei, Zeng, Li, 2013, *Towards Cross-Domain Learning for Social Video Popularity Prediction*, IEEE Transactions on Multimedia. [Link](https://www.microsoft.com/en-us/research/?p=273903) | Transfer from Twitter topics to YouTube videos, to explain sudden popularity | Earlier example of cross-platform learning. Different task and signal |

## Search only: open before using
- Sadeque, Bethard, 2019, engagement prediction survey, [arXiv:1907.05442](https://arxiv.org/abs/1907.05442).
- Bandari, Asur, Huberman, 2012, forecasting news popularity from content, ICWSM. Pre-publication contrast.
- Cao et al., 2019, coupled graph networks for popularity on diffusion graphs, [arXiv:1906.09032](https://arxiv.org/abs/1906.09032).
- Spina, Gonzalo, Amigó, about 2013, filter keywords for company name disambiguation on Twitter. Method for plan item D. [Link](https://research-repository.rmit.edu.au/articles/journal_contribution/Discovering_filter_keywords_for_company_name_disambiguation_in_twitter/27516885)
- Grinsztajn et al., 2022, trees against deep learning on tabular data, NeurIPS. In-distribution only.
- Jonkers, Van Wallendael, 2024, conformal predictive systems under covariate shift, [arXiv:2404.15018](https://arxiv.org/abs/2404.15018).
- Minimum-risk recalibration of classifiers, NeurIPS 2023. Few target labels; classifiers only.
- Wedel, Ohme, 2025, and Chouaki et al., 2024: data donation. They capture what a user saw, not a post's total views.
- Pierri et al., 2025, DSA research access, [arXiv:2512.14223](https://arxiv.org/abs/2512.14223); LoudCrowd help page; SocialCrawl engagement benchmarks.

## What the scan suggests
- No source found contradicts our main results. This is a quick scan, so it does not show that none exists.
- No peer-reviewed views-per-reaction ratio turned up, nor any validation of reach estimates against true views. These may be gaps worth stating, after a more careful search.
- No comparison of ridge and boosting under platform shift turned up. Present our result as empirical.
- No public dataset with real Facebook or Instagram views turned up, which supports collecting our own sample (plan item A).
