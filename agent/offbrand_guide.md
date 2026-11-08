# Off-brand mention guide (E224)

A codebook for one question about one post: **is this post really about the brand it was matched to?**

Brand24 finds mentions by matching keywords, and some brand names are also ordinary words, places, people or other
companies. A mention can therefore be recorded for a brand while the post is about something else entirely. This guide
is used for the 300-post validation sample in `data/processed/offbrand_sample.csv`; the sample is blind, so neither the
guide nor the coder may look at detector scores.

## Labels

### `about_brand`
The brand, or something the brand sells, is the **subject** of the post. That includes:
- the company named directly, in any tone: praise, complaint, a question about a service, news about the company;
- a product or a product line of the brand, even when the corporate name is not there (a beer brand owned by a brewer,
  a shoe label owned by a fashion group, a game studio owned by a publisher);
- the brand's shops, staff, prices, adverts, stock, sponsorship or an event it runs;
- an official account posting about itself;
- a second-hand or resale listing of a branded item: the item and its brand are what the post is about;
- a post whose author works, or worked, for the brand and talks about that experience.

### `not_about_brand`
The matched keyword is doing different work in the post. Typical cases:
- another meaning of the word: the sport, the insect, a weather word, a common noun or verb;
- a place, a person, a different company, or a different product with the same name;
- the brand is one item in a plain list (payment providers, tools, phone networks) where the post makes no statement
  about it, or the keyword is a generic phrase that happens to contain the brand's words;
- the post is about a *different* brand's product and the matched name only appears in a passing comparison.

### `unclear`
Use this rather than forcing a choice when the text does not let you decide:
- the visible part is only masked handles, emoji, a timestamp, or a fragment cut off before any subject appears;
- the language or the transliteration is too uncertain for you to follow the sentence;
- the name is used for something that could plausibly be the brand's (a venue, an award, a fan account) but the post
  does not say which, and the rest of the text is uninformative;
- the post mentions the brand's product only as an accessory to another subject, so a careful reader could go either way.

`unclear` is not a dumping ground: if a reasonable reader would answer the question from this text, pick one of the two
real labels.

## How to decide

1. Read the text and say in one sentence what the post is about. If that sentence is about the brand, its product or its
   business, the label is `about_brand`.
2. If the sentence is about something else and the brand word got in as a different meaning, `not_about_brand`.
3. Judge only the text you are given (it is cut at 600 characters) and the brand name in the row. Do not use the brand's
   industry as evidence, do not guess from the platform, and do not try to look the post up.
4. A hashtag or a tag is weak evidence: many posts carry a pile of tags that have nothing to do with the sentence. A tag
   alone, with no sentence about the brand, is not enough for `about_brand`.
5. When a post talks about a parent company's product, that is `about_brand`, even if the corporate name never appears.
6. When two readings are genuinely possible and the text is short, choose `unclear`.

## Boundary cases (my own words, no post quoted)

1. **A building described by a phrase that is also a clothing brand.** A history caption explains how an aircraft once
   hit a skyscraper and mentions the tower's upper side. The words of an outdoor clothing brand appear only as that
   ordinary phrase. The post is a historical anecdote, not about the clothing company: `not_about_brand`.

2. **An ordinary phrase that is also a shop's name.** A gadget review's title contains two very common words that also
   form an electronics chain's name, and the review is about a water heater. Nothing in the post discusses the chain:
   `not_about_brand`. This case shows why a keyword match can be technically correct and substantively wrong.

3. **An employee remembering the workplace.** Someone describes how cashiers at a homeware chain were pushed to ask
   customers for donations. It is not promotional and not positive, but the company's practice is exactly what the post
   is about: `about_brand`.

4. **A second-hand item with a visible logo.** A resale listing describes the size, colour and condition of a branded
   cap. The post exists because of the branded product: `about_brand`, even though the brand did not write it and the
   tone is commercial.

5. **A venue named after a brand.** A ticket request for a concert at an arena that carries a clothing label's name,
   where the matched brand is the label's parent group. The post is about the concert and the venue name is a place
   name; nothing is said about the group. I mark this `unclear` rather than `not_about_brand`, because the venue is a
   lasting trace of the brand and the text does not settle it.

6. **Almost no text.** A reply made of masked handles and two words with no subject at all: `unclear` — there is nothing
   to judge, and guessing would add noise to the validation.

## Notes for the team

- The labels produced for this sample are **AI labels from three independent passes plus an adjudication**; they are a
  stand-in until a team member checks a part of the sample. The card `agent/vault/E224.md` says so.
- Report the label as one of the three exact strings, one row per `sample_id`.
