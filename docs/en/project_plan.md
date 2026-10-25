# Project plan: sessions 3–8

Status after M1: question, hypotheses H1–H3, reconstructed Brand24 rules and a preliminary feasibility test are done (see `m1_exploration_card.pdf`).

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
