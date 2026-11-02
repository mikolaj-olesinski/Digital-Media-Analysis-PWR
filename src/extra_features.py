"""Extra feature groups for notebooks/models/04_extra_features.ipynb: simple NLP, lexicon sentiment, calendar, ratios, brand size.
All are label-free (use no views), so they may be computed on all mentions; the SVD of TF-IDF is fitted on the platform's text only."""
import re
import numpy as np, pandas as pd, holidays
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import eda_utils as eu

URL, TAG, MENT = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"#\w+"), re.compile(r"@\w+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
PROMO = re.compile(r"\b(sale|discount|offer|deal|coupon|promo|buy|free|giveaway|code|shop|price|order)\b", re.I)


def full_text(raw):
    return (raw.title.fillna("") + " " + raw.content.fillna("")).str.strip()


def text_stats(t):
    n = t.str.len().clip(lower=1)
    return pd.DataFrame({"chars": np.log10(n + 1), "words": np.log10(t.str.split().str.len().fillna(0) + 1), "n_urls": t.map(lambda s: len(URL.findall(s))),
                         "n_hashtags": t.map(lambda s: len(TAG.findall(s))), "n_mentions": t.map(lambda s: len(MENT.findall(s))), "n_emoji": t.map(lambda s: len(EMOJI.findall(s))),
                         "caps_share": t.map(lambda s: sum(c.isupper() for c in s)) / n, "digit_share": t.map(lambda s: sum(c.isdigit() for c in s)) / n,
                         "n_questions": t.str.count(r"\?"), "n_exclaim": t.str.count("!"), "promo_words": t.map(lambda s: len(PROMO.findall(s))),
                         "non_ascii_share": t.map(lambda s: sum(ord(c) > 127 for c in s)) / n})


def vader(t):
    a = SentimentIntensityAnalyzer(); sc = t.map(a.polarity_scores)
    return pd.DataFrame({"vader_compound": sc.map(lambda d: d["compound"]), "vader_neg": sc.map(lambda d: d["neg"]), "vader_pos": sc.map(lambda d: d["pos"])})


def tfidf_svd(t, k=30, seed=42):
    X = TfidfVectorizer(min_df=3, max_features=30000, sublinear_tf=True).fit_transform(t.where(t.str.strip() != "", "emptytext"))
    Z = TruncatedSVD(k, random_state=seed).fit_transform(X); return pd.DataFrame(Z, index=t.index, columns=[f"svd{i}" for i in range(k)])


def calendar(raw):
    ts = raw.created_at; hol = holidays.country_holidays("US"); dates = ts.dt.date
    return pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.weekday, "weekend": (ts.dt.weekday >= 5).astype(int), "us_holiday": dates.map(lambda d: d in hol).astype(int),
                         "hours_to_metrics": (raw.metrics_updated_at - ts).dt.total_seconds() / 3600})


def ratios(f):
    L, C, S, F = (10 ** f.log_likes - 1), (10 ** f.log_comments - 1), (10 ** f.log_shares - 1), 10 ** f.log_followers
    return pd.DataFrame({"likes_per_follower": np.log10(L / F + 1e-6), "comments_per_like": C / (L + 1), "shares_per_like": S / (L + 1), "interactions": np.log10(L + C + S + 1)})


def brand_size(meta):
    return pd.DataFrame({"brand_mentions": np.log10(meta.brand.map(meta.brand.value_counts()))}, index=meta.index)


def build(root, f, meta):
    """Returns {group name: DataFrame aligned to f.index}. Text SVD is fitted per platform."""
    raw = eu.load(root).astype({"mention_id": str}).loc[f.index]; t = full_text(raw)
    svd = pd.concat([tfidf_svd(t[meta.platform == p]) for p in meta.platform.unique()]).reindex(f.index)
    return {"text stats": text_stats(t), "VADER sentiment": vader(t), "TF-IDF SVD (30)": svd, "calendar": calendar(raw), "engagement ratios": ratios(f), "brand size": brand_size(meta)}
