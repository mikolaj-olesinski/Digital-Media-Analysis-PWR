# Talking points for the class slides

Working text for `pitchdeck/class_en.pdf`. Not preprint text. Say it in your own words. The fuller Polish study version is built by `pitchdeck/study_build.py`.

## Slide 1: what the data show
Platforms often hide how many people saw a post, and Brand24 still has to estimate it. We have 12,500 posts from five platforms. True views exist only on TikTok, YouTube and part of X, so that is where we test what works.

"Off by ×3" means the estimate is typically 3 times too high or too low. A constant times followers is off 7-8 times. An ordinary tree model with reactions, even without tuning, is off 2-3 times. On X, Brand24's reach is off about 12 times and our model about 3 times.

With one piece of information, likes do best. On TikTok likes alone give about ×2. The type of post alone says almost nothing. Our AI labels barely improve the estimate, but they show that at the same reactions promo gets 1.5-1.9 times the views.

## Slide 2: models and approaches
We tested 10 kinds of model, from plain ridge through forests to four boosters and a small neural network, each with 40 sets of settings chosen without looking at the test data. SVR on TikTok and CatBoost on YouTube and X came out best: about ×1.9, ×2.8 and ×3. Differences between good models are small.

We also added text, calendar, labels and graph features. Text gives a small gain on YouTube and X, the rest almost nothing. Removing likes hurts most.

## Slide 3: another platform and calibration
Facebook and Instagram have no views, so we check whether a model works on a platform it has not seen. Plain ridge on likes and comments works best. TikTok and YouTube are nearly as good as on their own data: ×2.1 and ×4.1. X gives ×4.5 instead of ×3.3: the model ranks posts well but overestimates them all about 2.5 times.

Calibration shifts all predictions by the typical error on a few posts with true views. On X about 50 posts cut the error from ×4.5 to ×3.5, close to perfect calibration. Brand24 has no such Facebook or Instagram sample, so we plan to collect our own. Those two platforms still have only an unchecked estimate.

## Slide 4: graphs
Each post is a point, linked to its 10 most similar texts. The Louvain algorithm splits this graph into groups with no views or labels. 13 variants of graph features give practically no gain for the estimate, and label propagation hurts.

Graphs do describe the data. Industries emerge by themselves: cars, hotels, fast food. On X, 38% of near-identical posts are spam, against 9% of the rest. A curiosity: mentions of the Cricket Wireless network on TikTok are about the sport, 0 of 20 about the company. It is a small sample, so we want to measure how often this happens.
