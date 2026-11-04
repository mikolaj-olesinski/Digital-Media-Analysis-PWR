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
| E36 | instrument, unchecked | Brand industry labels | [E36](E36.md) |
| E37 | finding, needs hand check | Brand away from its industry in the text graph | [E37](E37.md) |
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
| E55 | finding | Leakage audit: folds that also separate authors and copied texts | [E55](E55.md) |
| E56 |  | Noise ceiling: posts with almost the same inputs | [E56](E56.md) |
| E57 | small | Quantile models and tail weighting | [E57](E57.md) |
| E58 | finding | Abstention: error on the posts kept | [E58](E58.md) |
