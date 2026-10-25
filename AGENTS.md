# AGENTS.md: AMC project (media analysis)

Instructions for AI agents working in this repository.
Read in full before every session. Course rules: `docs/pl/course_rules.pdf` (in Polish).

## 1. Project topic

> **Topic (from the instructor's list): "How to estimate the visibility of social media posts"**

### Business context (from the client, Brand24)
- Social media platforms **often do not disclose the real visibility (reach/views) of posts**,
  partly because they earn money from ads (AdSense and equivalents) and reach is a commercial
  asset for them.
- **Brand24** monitors brand mentions and wants to **estimate** visibility where the platform
  does not provide it (Facebook, Instagram, partly X).
- Today Brand24 uses a **simple constant: reach ≈ constant × the author's followers**
  (field `b24_estimated_reach`). The client gave the team information about this constant:
  **value(s) of the constant: _[TO FILL IN from the instructor/Brand24]_**.
- Research goal: can visibility be estimated **better than by a constant** (e.g. from followers,
  engagement: likes/comments/shares, platform, language, time) and **with what uncertainty**;
  where we have hard data (TikTok, YouTube: `view_count`) it serves as reference data
  (ground truth), and from it we judge whether the approach can be transferred to platforms
  without views. An honest negative result ("on FB/IG this cannot be validated") is fully
  acceptable.

Data: `data/raw/reach_samples_students.csv`: 12,500 mentions (2,500 each from X/Twitter, Instagram,
YouTube, Facebook, TikTok) with columns `view_count`, `author_followers`,
`likes/comments/shares` and `b24_estimated_reach`. Reach is computed **differently depending on the
platform** (see §3).

Measure of success according to the course: **a reliable, verified conclusion whose scope and certainty
match the evidence**. A negative result (no effect / impossible to measure) is a full result.
The model's score alone (e.g. F1) does not count.

## 2. Course rules that bind the agent

- Final grade: 40% preprint (≤8 pages, session 5), 20% poster (session 7), 20% milestones
  M1–M3, 20% activity. Points are scaled by individual contribution.
- **Graded**: commit history, **AI agent session logs**, the pangram test (whether the publication
  text is generated), individual defence.
- Hence:
  1. **Log every session** in `logs/YYYY-MM-DD_<person>_<agent>.md` (goal, prompts, what
     changed, decisions). Do not delete logs.
  2. **Do not generate the final text of the preprint/poster.** Help with analysis, code, figures
     and critique; the text is written by people in their own words. Every use of AI is disclosed in the preprint.
  3. Commits are small, descriptive, from a specific person. The number of commits ≠ contribution, but their absence = no trace.
  4. Everything must be reproducible: script/notebook from raw data to figure.
- Data: compliance with **GDPR** and platform terms. Do not extract or publish personal data
  (`author_name`, `author_url`, `url`) beyond what is necessary; aggregate or pseudonymise in
  reports. No private authors' names on the poster/preprint.
- Do not fabricate numbers, citations or results. Show uncertainty (intervals, bootstrap).

## 3. Data: what we already know (preliminary EDA, 2026-10-12)

`data/raw/reach_samples_students.csv`: 12,500 rows × 19 columns, 348 brands, mostly EN (85%), country
often empty (53%). Dates: 2025-01-01 → 2026-10-10, but **~85% of rows from April–August 2026**
(earlier ~250–330/month, from 2026-04 ~1,000–1,900/month) → the sample is not uniform in time.
No duplicate `mention_id` or `url`.

| platform | `view_count` | `author_followers` | `shares` | reach == views | reach = 0 |
|---|---|---|---|---|---|
| tiktok | 100% | 100% | 100% | **100%** | 0% |
| youtube | 99% | 100% | 59% | **99%** | 2% |
| twitter | 75% | 100% | 100% | 1% | 11% |
| facebook | **0%** | 73% | 100% | – | 9% |
| instagram | **0%** | 8% | **0%** | – | **43%** |

Working conclusions (to be verified, not to be cited as fact):
- TikTok and YouTube: reach ≈ number of views (a "hard" measure).
- Facebook/Instagram: no views → reach is **modelled/estimated** (median reach/followers
  about 6% FB, 3% IG); Instagram has 43% zeros.
- Twitter: reach ≠ views and ≠ followers; it looks like an estimate (median reach/followers ≈ 10%).
- **Brand24 constant visible in the data** (reach / followers, only rows with followers>0 and reach>0):
  - Twitter: median ≈ **0.10** (most common value 0.1; ~75% of rows ≤ 0.1; the rest are exceptions,
    probably posts with real views, to be checked).
  - Facebook: a **stepwise constant, decreasing with account size**: about 0.10 (small accounts) → 0.06 →
    0.05 → 0.04 → 0.02 (largest). So not one constant but thresholds.
  - Instagram: no clear constant (only 8% of rows have followers, 43% zeros).
  - To be verified on the full data and compared with information from Brand24.
- Distributions are strongly right-skewed (median reach 366, mean 15,652, max ≈ 16 million) → use
  log scale, medians, quantiles; not means.
- Missing data: `sentiment` 42% NaN, `title` 20%, `author_*` 24%, `view_count` 45%.
  Missingness is **not random** (depends on platform): report it per platform.
- Types: `author_followers`, `view_count`, `likes`… are `float` because of NaN.

## 4. Team roles (3–4 people; fill in names)

| Role | Person | Responsibility |
|---|---|---|
| Research lead | _TBD_ | question, hypotheses, consistency of the conclusion |
| Data engineer | _TBD_ | data acquisition, GDPR/terms, *data statement* |
| Pipeline engineer | _TBD_ | codebook, gold standard, instrument validation, error analysis |
| Analyst engineer | _TBD_ | robustness analyses, visualisations, A5 (in a team of 3 combines lead + metrologist) |

The agent supports the role of the person who started it, and logs which one.

## 5. Step-by-step plan

### Sessions 1–2 → **M1: data exploration card**
The card must contain: research question + hypothesis, data sources, sample scope, preliminary
exploration results, measurement plan, data limitations, possible sources of error, a justification of
feasibility at the course scale. Graded: concreteness of the question, quality of exploration, consistency of the plan.

1. **Confirm the topic** from the instructor's list; adjust §1 if needed.
2. **Team and roles** (§4); `git init`, first commit, `logs/`.
3. **Refine the question and hypotheses** (falsifiable), e.g.:
   - H1: on TikTok/YouTube reach = views (direct measure).
   - H2: on FB/IG/X reach is a function of followers/engagement; check how well.
   - H3: the error/uncertainty of the estimate differs between platforms.
   - For each hypothesis: what result would refute it?
4. **EDA** (notebook `notebooks/01_eda.ipynb` or `src/eda.py`):
   missingness per platform, distributions (log), reach vs views / followers / engagement,
   zeros, outliers (max 16 million!), distribution in time (jump from 2026-04), brands/languages/countries,
   duplicates and near-duplicates of content, bots/spam (e.g. coupon offers, "AI screener").
5. **Measurement plan**: how do we measure "reach reliability" without ground truth? (e.g. agreement with
   `view_count` where it exists; regression/rule reconstruction; stability; effect of metrics_updated_at).
   State that true reach on FB/IG is unknown → the scope of conclusions is limited.
6. **M1 card** (≤2 pages, `docs/en/m1_exploration_card.*` + `docs/pl/` version) + 2–4 key figures.
7. Plan the next stages and the division of work up to session 3.

### Session 3: codebook and gold standard; instrument validation
Define the coding (e.g. mention type: organic / ad / spam / bot), hand-label a random stratified
sample (gold standard), compute coder agreement (kappa), validate the instrument on reference data.

### Session 4: first result on the full data
Run the pipeline on everything, first estimates with uncertainty, error analysis.

### Session 5: **M2: red teaming of the report** (+ preprint ≤8 pages)
A version of the report with methodology and results; tournament: raising and defending against remarks.
Prepare a list of your own weak points **before** the tournament; do not repeat remarks already used.

### Session 6: consultations · Session 7: **M3** (poster) · Session 8: poster session
The poster = the same content as the preprint in a presentation layer: information hierarchy, readable
figures with uncertainty, concise language, fidelity of the message. 8-minute presentation + individual defence.

## 6. Repository conventions

```
AGENTS.md                       # this file (local only, git-ignored)
README.md                       # project overview
docs/pl/course_rules.pdf        # course rules (do not modify)
data/raw/reach_samples_students.csv   # raw data (git-ignored): NEVER edit, only read
data/processed/                 # derived data (generated by scripts)
notebooks/  src/                # analyses
figures/                        # figures (generated, with captions and axes), *_en.png / *_pl.png
docs/en/, docs/pl/             # M1 card, codebook, data statement, preprint: one folder per language
pitchdeck/en/, pitchdeck/pl/     # presentation per language; build.py and strings_en.py in pitchdeck/
logs/                           # agent session logs (mandatory, git-ignored)
```

- Python + pandas; randomness with a fixed `seed`. Documents are kept in two language versions
  (separate `en/` and `pl/` folders with the same file names); code and comments in English.
- Commit messages are written in English, short and descriptive. Do not add AI agents as co-authors or contributors in commits.
- Figures: labelled axes, units, legend, uncertainty; log scale for reach.
- Before changing data/scripts: `git status`; do not overwrite other people's work.

## 7. What the agent does NOT do

- Does not write the final text of the preprint/poster or "improve" it so as to hide the use of AI.
- Does not modify the raw CSV. Does not send data to external services without the team's consent.
- Does not draw causal conclusions from correlations; does not overstate the certainty of results.
- Does not report model metrics as a goal in themselves: the conclusion and its scope are what count.
