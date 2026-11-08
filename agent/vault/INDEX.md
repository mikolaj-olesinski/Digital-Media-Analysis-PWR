# Experiment vault

One card per experiment. Generated index; do not edit by hand (run `python src/agent_tools.py vault`).

| ID | Type | Question | Card |
|---|---|---|---|
| L1 | finding | Can AI label type and sentiment? | [L1](L1.md) |
| E01 | gain | Which column groups matter? | [E01](E01.md) |
| E02 | finding | Do intervals hold? Does a model transfer? | [E02](E02.md) |
| E03 | none | Do other models or first graph features help? | [E03](E03.md) |
| E04 | small | Labels vs views | [E04](E04.md) |
| E05 | small | Do extra columns help? | [E05](E05.md) |
| E06 | small | Brand graph, predicting reactions | [E06](E06.md) |
| E07 | none | Graph lab | [E07](E07.md) |
| E08 | finding | What does the text graph find? | [E08](E08.md) |
| E09 | none | Do reactions transfer to FB/IG? Two-stage reactions then views | [E09](E09.md) |
| E10 | gain | Default model vs baselines | [E10](E10.md) |
| E11 | small | Does tuning change the ranking? | [E11](E11.md) |
| E12 | small | Tuned boosters with text | [E12](E12.md) |
| E13 | finding | Full transfer matrix | [E13](E13.md) |
| E14 | none | Tuning for transfer, label correction | [E14](E14.md) |
| E15 | gain | Calibration with few posts | [E15](E15.md) |
| E16 | finding | FB/IG sensitivity to source | [E16](E16.md) |
| E17 | finding | Three-way ablation with labels | [E17](E17.md) |
| E18 | finding | Systematic errors | [E18](E18.md) |
| E19 | finding | Promo vs views per reaction | [E19](E19.md) |
| E20 | none | Dropping spam from training | [E20](E20.md) |
| E21 | finding | Community detection | [E21](E21.md) |
| E22 | finding | False brand matches | [E22](E22.md) |
| E23 | small | YouTube transcripts | [E23](E23.md) |
| E24 | finding | Brand24 rule | [E24](E24.md) |
| E25 | gain | Rule vs model on X | [E25](E25.md) |
| E26 | small | One is-promo column | [E26](E26.md) |
| E27 | finding | Is promo behind YouTube errors? | [E27](E27.md) |
| E28 | none | Removing zero-reaction posts | [E28](E28.md) |
| E29 | small | Second graph round | [E29](E29.md) |
| E30 | none | Graph bridge across platforms | [E30](E30.md) |
| E31 | none | Two-stage models | [E31](E31.md) |
| E32 | gain | MAE loss | [E32](E32.md) |
| E33 | finding | Error by reaction tier | [E33](E33.md) |
| E34 | gain | Interval per reaction tier | [E34](E34.md) |
| E35 | finding | False match seen in the text graph | [E35](E35.md) |
| E36 | instrument | Brand industry labels | [E36](E36.md) |
| E37 | finding | Brand away from its industry in the text graph | [E37](E37.md) |
| E38 | finding | What is in the second Brand24 file? | [E38](E38.md) |
| E39 | finding | How fast do views grow? | [E39](E39.md) |
| E40 | gain | Forecast later views from the first check | [E40](E40.md) |
| E41 | finding | Forecast without views at the first check | [E41](E41.md) |
| E42 | finding | Interval of the forecast | [E42](E42.md) |
| E43 | gain | Estimate at posting time (no reactions) | [E43](E43.md) |
| E44 | small | Does the age of the view check help the final model? | [E44](E44.md) |
| E45 | worse | Are the newer checks more predictable? | [E45](E45.md) |
| E46 | finding | Rolling window: does the final model hold up when trained only on earlier months? | [E46](E46.md) |
| E47 | small | Do tabular foundation models (TabPFN v2, TabICL) beat or add to the final model? | [E47](E47.md) |
| E48 | finding | Classification: viral top 10%, and posts without a view count | [E48](E48.md) |
| E49 | none | Context from other posts of the same author, hashtag and brand (no view counts used) | [E49](E49.md) |
| E50 | small | Text: local sentence embeddings (multilingual-e5-small) and TF-IDF | [E50](E50.md) |
| E51 | small | Offset target: learn views per like | [E51](E51.md) |
| E52 | small | Posting-time estimate with sentence embeddings and context from other posts | [E52](E52.md) |
| E53 | small | Fine-tuned transformer: the row as text, and text as a correction | [E53](E53.md) |
| E54 | small | Stack of every model family on the same folds | [E54](E54.md) |
| E55 | finding | Leakage audit: folds that also separate authors and copied texts | [E55](E55.md) |
| E56 | finding | Noise ceiling: posts with almost the same inputs | [E56](E56.md) |
| E57 | small | Quantile models and tail weighting | [E57](E57.md) |
| E58 | finding | Abstention: error on the posts kept | [E58](E58.md) |
| E59 | finding | Forecast horizon curve; training on every pair of checks | [E59](E59.md) |
| E60 | gain | Early-reaction forecast without views (FB/IG setting) | [E60](E60.md) |
| E61 | worse | One pooled model for three platforms, monotone constraints | [E61](E61.md) |
| E62 | finding | Why YouTube views per like fell in Jul-Aug | [E62](E62.md) |
| E63 | finding | Copy-paste campaigns | [E63](E63.md) |
| E64 | small | Separate model for posts without likes and comments | [E64](E64.md) |
| E65 | finding | GraphSAGE on the text-similarity graph | [E65](E65.md) |
| E66 | small | Account type from the author name | [E66](E66.md) |
| E67 | small | Topic clusters of all posts | [E67](E67.md) |
| E68 | small | Author over-performance persists | [E68](E68.md) |
| E69 | small | Brand communities and cold start | [E69](E69.md) |
| E70 | small | Author history in depth, and at posting time | [E70](E70.md) |
| E71 | small | Can the author's level be predicted without history? | [E71](E71.md) |
| E72 | small | Variance components of the error | [E72](E72.md) |
| E73 | small | Brand totals | [E73](E73.md) |
| E74 | small | Anomalies ×10 above or below the prediction | [E74](E74.md) |
| E75 | small | Views per like implied by the final model | [E75](E75.md) |
| E76 | small | Where the remaining error sits | [E76](E76.md) |
| E77 | small | How many earlier author posts are needed | [E77](E77.md) |
| E78 | none | Author level without views | [E78](E78.md) |
| E79 | none | Live forecast plus author history | [E79](E79.md) |
| E80 | small | Intervals for brand totals | [E80](E80.md) |
| E81 | finding | FB / IG next to platforms with views | [E81](E81.md) |
| E82 | pending | Pre-registered FB / IG estimates | [E82](E82.md) |
| E83 | finding | Residual atlas | [E83](E83.md) |
| E84 | none | Country, local time, holidays, industry as columns | [E84](E84.md) |
| E85 | finding | Where promo over-performs | [E85](E85.md) |
| E86 | none | Brand popularity on Wikipedia | [E86](E86.md) |
| E87 | finding | Mapping reaction scales between platforms | [E87](E87.md) |
| E88 | small | Adapting the out-of-platform model to the target's reaction structure | [E88](E88.md) |
| E89 | small | Few-shot recalibration with k target posts | [E89](E89.md) |
| E90 | none | Anchor regression, group DRO, stabilised weights | [E90](E90.md) |
| E91 | none | Input subset by source-to-source transfer | [E91](E91.md) |
| E92 | none | Brand effect carried over from other platforms | [E92](E92.md) |
| E93 | finding | How wrong is the transfer to Facebook, measured? | [E93](E93.md) |
| E94 | worse | Do followers improve the measured transfer to Facebook? | [E94](E94.md) |
| E95 | finding | Does the FB transfer hold beyond the hand-check sample? | [E95](E95.md) |
| E96 | finding | How concentrated is reach? | [E96](E96.md) |
| E97 | small | Which FB posts to hand-check with a budget of k? | [E97](E97.md) |
| E98 | finding | Intervals for the FB estimate | [E98](E98.md) |
| E99 | finding | Where does Brand24's reach rule fail? | [E99](E99.md) |
| E100 | worse | Does text help the Facebook transfer? | [E100](E100.md) |
| E101 | small | Do views need age normalisation? | [E101](E101.md) |
| E102 | finding | Brand scorecard: ranking vs level | [E102](E102.md) |
| E103 | none | Are the same texts posted across platforms? | [E103](E103.md) |
| E104 | finding | Viral alert at the top 1% | [E104](E104.md) |
| E105 | finding | Is the X view count missing at random? | [E105](E105.md) |
| E106 | finding | Alert budget: recall at k alerts for the top 1% | [E106](E106.md) |
| E107 | finding | Brand portfolio: what would change on X? | [E107](E107.md) |
| E108 | finding | Confidence flags from metadata | [E108](E108.md) |
| E109 | finding | Calibrated brand totals with intervals on X | [E109](E109.md) |
| E110 | none | Does posting cadence reduce reach per post? | [E110](E110.md) |
| E111 | finding | Brand24's FB reach against measured views | [E111](E111.md) |
| E112 | finding | How many posts does a brand total need? | [E112](E112.md) |
| E113 | none | Multi-task across platforms | [E113](E113.md) |
| E114 | finding | Validation audit and post-hoc recalibration | [E114](E114.md) |
| E115 | finding | Calibration budget for a new platform | [E115](E115.md) |
| E116 | small | Are YouTube zeros missing? | [E116](E116.md) |
| E117 | finding | What is a 0/0 post at the first scan worth? | [E117](E117.md) |
| E118 | finding | Conformalized quantile intervals | [E118](E118.md) |
| E119 | small | Author target encoding as a feature | [E119](E119.md) |
| E120 | none | Cross-platform author history | [E120](E120.md) |
| E121 | none | Recency, sequence and text-surface columns | [E121](E121.md) |
| E122 | worse | Spectral embeddings of the author-brand graph | [E122](E122.md) |
| E123 | finding | CQR intervals for the FB transfer | [E123](E123.md) |
| E124 | small | Final estimator v2: author_te + CQR | [E124](E124.md) |
| E125 | none | CQR by reaction tier | [E125](E125.md) |
| E126 | finding | Range for a 0/0 post | [E126](E126.md) |
| E127 | worse | Metapath2vec on author-brand-hashtag | [E127](E127.md) |
| E128 | finding | Which graph edges carry signal? | [E128](E128.md) |
| E129 | none | Distributional model vs CQR | [E129](E129.md) |
| E130 | finding | Intervals at posting time | [E130](E130.md) |
| E131 | finding | Out-of-platform CQR matrix | [E131](E131.md) |
| E132 | finding | FB CQR by tier | [E132](E132.md) |
| E133 | none | Recency weighting against drift | [E133](E133.md) |
| E134 | none | Seed averaging | [E134](E134.md) |
| E135 | worse | Follower-tier recalibration | [E135](E135.md) |
| E136 | finding | Rolling-origin validation | [E136](E136.md) |
| E137 | finding | 50% likely range | [E137](E137.md) |
| E138 | finding | Robustness to missing reactions | [E138](E138.md) |
| E139 | small | Target definition: views vs views/day | [E139](E139.md) |
| E140 | none | Hurdle model for zero views | [E140](E140.md) |
| E141 | small | Final X model v3 with log(age) | [E141](E141.md) |
| E142 | finding | Temporal CQR | [E142](E142.md) |
| E143 | worse | FB transfer with age | [E143](E143.md) |
| E144 | none | FB tier-specific CQR widths | [E144](E144.md) |
| E145 | none | FB tier calibration, pooled splits | [E145](E145.md) |
| E146 | none | Widening the FB band for >1M | [E146](E146.md) |
| E147 | finding | Can the >1M FB tier be calibrated? | [E147](E147.md) |
| E148 | finding | FB band excluding the largest pages | [E148](E148.md) |
| E149 | none | Followers as a feature for large FB pages | [E149](E149.md) |
| E150 | small | Dedicated model for the largest FB pages | [E150](E150.md) |
| E151 | none | Large-page model for the >1M FB band | [E151](E151.md) |
| E152 | finding | Error by brand size and industry | [E152](E152.md) |
| E153 | small | Hand-check sample vs full fetched sample | [E153](E153.md) |
| E154 | small | Hybrid FB model | [E154](E154.md) |
| E155 | none | Is the hybrid FB gain significant? | [E155](E155.md) |
| E156 | finding | FB error by industry | [E156](E156.md) |
| E157 | finding | FB brand totals and hybrid ranking | [E157](E157.md) |
| E158 | small | FB brand totals after calibration | [E158](E158.md) |
| E159 | small | FB brand-total interval scaled by n | [E159](E159.md) |
| E160 | none | FB brand interval with >=10 posts | [E160](E160.md) |
| E161 | finding | Robustness to missing followers | [E161](E161.md) |
| E162 | finding | Which signal does the FB transfer use? | [E162](E162.md) |
| E163 | finding | Which model when data is missing? | [E163](E163.md) |
| E164 | finding | Partial missingness crossover | [E164](E164.md) |
| E165 | finding | FB rule vs model by follower tier | [E165](E165.md) |
| E166 | small | FB intervals on the 130 hand posts | [E166](E166.md) |
| E167 | finding | How much do views depend on followers vs reactions per platform? | [E167](E167.md) |
| E168 | finding | How much of Brand24's reach rule can be recovered? | [E168](E168.md) |
| E169 | finding | Can brand history substitute for unknown author history? | [E169](E169.md) |
| E170 | none | Brand news attention on Wikipedia at post time | [E170](E170.md) |
| E171 | none | Company facts from Wikidata | [E171](E171.md) |
| E172 | none | Brand momentum, past-only | [E172](E172.md) |
| E173 | none | Brand buzz in the Brand24 stream | [E173](E173.md) |
| E174 | none | Platform time | [E174](E174.md) |
| E175 | small | New text cues | [E175](E175.md) |
| E176 | finding | Estimator for whatever fields arrive, with calibrated confidence | [E176](E176.md) |
| E177 | finding | Facebook on Brand24's own likes / comments | [E177](E177.md) |
| E178 | small | FB result on few brands | [E178](E178.md) |
| E179 | finding | Check age in the noise ceiling; 30-day target | [E179](E179.md) |
| E180 | finding | Losses for totals | [E180](E180.md) |
| E181 | finding | X missing views, inverse-probability weights | [E181](E181.md) |
| E182 | finding | Error by language and sentiment | [E182](E182.md) |
| E183 | small | Creator score stability | [E183](E183.md) |
| E184 | small | Anomalies vs promo / spam | [E184](E184.md) |
| E185 | finding | Early virality alert from the first check | [E185](E185.md) |
| E186 | finding | Monthly brand ranking within an industry | [E186](E186.md) |
| E187 | finding | Negative posts | [E187](E187.md) |
| E188 | small | Earned media value | [E188](E188.md) |
| E189 | finding | Can we fetch an external YouTube/X/TikTok sample, and does the model hold on it? | [E189](E189.md) |
