# Project plan: sessions 3–8

Status after M1: question, hypotheses H1–H3, reconstructed Brand24 rules and a preliminary feasibility test are done (see `m1_exploration_card.pdf`).

**What happened to these items:** `plan_status.md` maps every idea (A-I) to its outcome. This plan is the plan as of
session 3 and is not updated.

Key caveat: on Facebook and Instagram there is no ground truth and `b24_estimated_reach` is Brand24's own rule. We therefore (1) learn views ~ engagement on TikTok, YouTube and X, (2) measure how much that relation differs between platforms, and (3) use that difference as the uncertainty of transferring to FB/IG. An honest negative result is acceptable.

## Session 3: codebook, gold standard, instrument validation
- [x] Codebook for mention type: organic / promo / spam / unclear and sentiment (`docs/en/codebook.md`)
- [x] Manual spot check of the AI labels by a team member (informal, not scored; no human-AI kappa)
- [x] Agreement between three AI coders (kappa 0.77-0.79 type, 0.71-0.73 sentiment); human-AI agreement not quantified
- [x] AI instrument labels all 12,500 mentions (3 Haiku coders + Sonnet adjudication); no human gold sample
- [ ] Check Brand24's `sentiment` field (42% missing) against the gold sample
- [x] Data statement and GDPR note (`docs/en/data_statement.md`)
- [x] Shared validation module `src/validation.py`: brand-grouped and temporal split, MAE in log10, bootstrap CI, conformal intervals with coverage
- [ ] Ask Brand24 to confirm the reconstructed rules

## Session 4: first result on the full data

_Done 2026-11-02:_ notebook `03_zoo_strata_graph` (model zoo, graph features, strata, labels vs views): none beats followers + engagement; see `logs/2026-11-02_mikolaj_claude.md`.

_Also done 2026-11-02:_ nested hyperparameter search over ten families (`07_tuning`), full out-of-platform matrix with calibration (`08_transfer_matrix`), three-way ablation with labels, bias, communities (`09_ablation_labels_bias`), reconstruction of the Brand24 rule (`10_brand24_rule`). Overview: `docs/en/analysis_overview.md`. Not done from this ladder: sentence embeddings, full GNN, multi-task networks, distributional models.

Model ladder, all evaluated with the same validation, per platform:
1. Baselines: constant × followers, the reconstructed Brand24 rule
2. Log-log regression, splines, quantile regression, Ridge/Lasso
3. Random Forest, ExtraTrees, kNN; gradient boosting (LightGBM, XGBoost, CatBoost); hurdle model for zeros; distributional models (NGBoost, CatBoost uncertainty)
4. Text: handcrafted features, multilingual embeddings (PCA/UMAP), content clusters, spam/duplicate detection
5. Graphs: author–brand bipartite graph features, content-similarity kNN graph (features or GNN), brand graph; neighbour features computed on training data only
6. Neural networks: MLP with quantile or Student-t head, multi-task (views, likes, comments, shares), optionally a fine-tuned small multilingual transformer
7. Intervals: quantile regression + conformal calibration, coverage reported per platform

Transfer and FB/IG:
- [ ] Ratio views / engagement per platform (TikTok, YouTube, X) and its stability
- [x] Leave-one-platform-out (`notebooks/models/02_transfer_and_intervals.ipynb`): TikTok and YouTube transfer well, X biased, cross-platform intervals under-cover
- [ ] Engagement model on FB and IG (likes, comments, shares from followers, text, brand, time) as a separate analysis
- [ ] Shared model with platform as a feature / domain adaptation
- [ ] Results table: what improves on the baseline (bootstrap CI) and what does not
- [ ] Error analysis: viral posts, zeros, missing followers

## Ideas to stand out (planned, not started)

Noted on 2026-11-02. Nothing here has been run. Brand24 told us they have no Facebook or Instagram sample with real views, so a check on those platforms needs data we collect ourselves.

Priority for the red teaming: A + B as the main line, D as the highlight, then E and F. Target claim, if the data allow it: "we tested on real FB/IG data what could not be tested before, and we show how much data a credible estimate needs". A negative result counts.

**A. Our own gold sample from Facebook and Instagram**
- Manually record public play counts on Reels and Facebook videos: 100-200 posts with views, likes, comments, followers. No scraping and no author names. Check platform terms and GDPR first, and record the decision in the data statement.
- Ask friends, student clubs, the university and small firms for post statistics from their own dashboards. Only this source covers ordinary posts and photos, and gives reach next to views, which answers the reach-vs-views question.
- Prepare a collection sheet and a sampling protocol before collecting, so the sample is not biased.
- Limits: Reels and videos are not ordinary posts, so conclusions cover that format only; friends' accounts are not representative.
- This is the first possible test of the model and of Brand24's rule on FB/IG, for example whether Facebook's 100 per reaction is too high.

**B. Calibration recommendation**
- Notebook 08 already has the curve on TikTok, YouTube and X. With A it becomes concrete for Brand24: "collect N posts from clients and the error falls from A to B". On FB/IG it stays only a hint until A exists.

**C. Intervals and abstention**
- The model gives an interval and says "I don't know" when a post is uncertain, for example with no reactions. Result: a curve of error against the share of posts kept.

**D. False brand matches**
- Community detection found "Cricket Wireless" mentions about the sport and "Aon" as a vlog hashtag. So far this is one example. First measure how often it happens, then build a detector and estimate what share of a brand's estimated reach such mentions make up. Needs no views, so it also runs on FB/IG.
- Needs 100-200 hand-labelled examples, which would also give the human gold standard we lack. Prepare a labelling sheet first.

**E. Community as a bridge to FB/IG**
- Build the text graph across all five platforms, so a Facebook post gets neighbours from TikTok or YouTube with real views. Test on the known platforms: does the level shift shrink when moving between TikTok, YouTube and X? Detect communities without views and test on brands outside training.
- Variant: one shared model plus a small correction per community (hierarchical), asking whether reactions to views differs between groups, rather than whether MAE falls. Separate models per community are unlikely to help: models per follower tier were worse than one shared model, and community features gained at most 0.007.

**F. Smaller tests**
- Posts without reactions: a separate model on text, author and community, where reactions say nothing and the error is largest.
- Virality classification: will a post reach the top 10%? Precision and recall on brands outside training.
- Copy-paste campaign detector: near-identical posts have view correlations of 0.66-0.82 and spam is 2-4 times more common among them. Do they inflate a brand's reach?

**G. Improve the extremes (user priority, 2026-11-02)**
- Error is largest at both ends of true views: lowest fifth 0.34 / 0.60 / 0.68 and top fifth 0.34 / 0.55 / 0.68, against 0.21 / 0.34 / 0.28 in the middle (TikTok, YouTube, X). The model overestimates small posts and underestimates virals. Posts with no reactions have the largest error by reaction count: 0.53 / 0.68 / 0.48. Source: `data/processed/ablation_numbers.json`, notebook 09.
- To test: quantile models (10%, 50%, 90%) instead of one number; post-hoc recalibration of predictions; weighting the tails in the loss; a separate model for posts with no reactions (see F); virality classification (see F); abstention (see C).
- Report per fifth of true views and per reaction tier, and also the error in absolute views, since a factor of 4 means little on small posts and a lot on virals.
- Caution: part of this pattern comes from splitting by the true value, so any imperfect model shows it. Spreading predictions usually raises overall MAE, so report the trade-off. Choose by what Brand24 needs.

**H. YouTube zeros (question for Brand24)**
- 23% of YouTube posts have zero likes and comments and hold 35% of the error. Of these, 29% have at least 1,000 views, and 30 have at least 10,000. Ask Brand24 whether a zero can mean "not collected" or "hidden". If so, test treating such zeros as missing. Source: notebook 11.

**I. Report and interval by reaction tier (tested, ready to use)**
- Notebook 14: error falls with reactions on YouTube and X; a single interval width covers only 67-87% of zero-reaction posts; one width per reaction tier restores about 90% where calibration data suffice. Use this in the final model and on the poster.

**Related reading:** methods and sources for C, D and F are in `docs/en/literature_notes.md`.

**Not planned:** more models or a GNN. The signal is in reactions, so at most hundredths of MAE are left there.

## Session 5: M2, red teaming and preprint (≤8 pages)
- [ ] Report with methodology and results, written by people in their own words (AI use disclosed)
- [ ] List of our own weak points before the tournament
- [ ] Notes on received remarks, replies and a correction plan

## Session 6: consultations
- [ ] Questions for the instructor, fixes after red teaming

## Session 7: M3, poster
- [ ] Poster (same content as the preprint), note for a non-expert reader
- [ ] 8-minute presentation rehearsed

## Session 8: poster session and individual defence
- [ ] Everyone can explain the pipeline, their role and the limits of the conclusions
