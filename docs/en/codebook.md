# Codebook: mention type and sentiment

Used to label each mention with an LLM (see `src/make_label_chunks.py`). A mention is judged only from the
text shown (brand, platform, language, title/content truncated to 400 characters).

## type (exactly one)
| label | definition | typical signs |
|---|---|---|
| `organic` | a person writes or reacts in their own voice, about their own experience, opinion, question or conversation, whether or not the brand is the main topic | personal tone, replies to others, opinions, stories |
| `promo` | commercial or institutional communication: the brand's own announcements, ads, sponsored posts, influencer promotion, news or marketing from a company or media outlet | calls to action, "link in bio", product launches, press-release style, discount of the brand itself |
| `spam` | content unrelated to the brand's real topic or mass-produced: coupon/deal farming, "AI screener" style offers, scams, adult content, keyword stuffing, repeated templates, giveaways asking for tags | lists of unrelated brands or hashtags, "promo code", "click here", templated text |
| `unclear` | too short, empty, or ambiguous to decide (e.g. only emojis, only a link, a single word) | use sparingly |

Tie-break: if a post is a person promoting something for money, use `promo`; if it is mass-produced and unrelated to the brand, use `spam`; otherwise `organic`.

## sentiment (exactly one, about the text as a whole)
`positive`, `neutral`, `negative`. Use `neutral` for facts, questions and mixed or unclear tone.
