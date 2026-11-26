

This notebook takes a poem and replaces every **content word** (noun, verb, adjective, adverb) with its **closest semantic associate** from a large pretrained embedding model. Function words, punctuation and the **grammatical form** of each word stay the same. Then it looks at how the meaning changes.

The task is done twice:
1. **English**: *The Road Not Taken* by Robert Frost (1916)
2. **Dari/Persian**: the opening of Rumi's *Masnavi* (نی‌نامه, "Song of the Reed"), done as an extra experiment in my own language

## Pipeline

| Step | English | Dari / Persian |
|---|---|---|
| 1. Model | `fasttext-wiki-news-subwords-300` (300d, ~1M words, ~16B tokens) | fastText `cc.fa.300`, top 200k words streamed (~130 MB instead of 1.26 GB) |
| 2. Preprocessing | lowercase, remove punctuation, split | normalise Arabic/Persian letters (ي→ی, ك→ک), remove diacritics, keep ZWNJ and ezafe |
| 3. Content words | spaCy POS tagging in context (`NOUN`, `VERB`, `ADJ`, `ADV`) | no Persian POS tagger, so function words are excluded instead: spaCy stop words plus archaic poetic words (کز، کاندر، لیک، ز …) |
| 4. Associates | `most_similar(word, topn=1)` plus a filtered top-1 that skips the same word in another form or spelling | same idea, plus frequency filtering to remove Common Crawl junk tokens |
| 5. Keep the form | re-inflect with **lemminflect** (`walked` → `turned`, `roads` → `highways`) | rule-based morphology: plural ‌ها/ان, verb prefixes ب/ن/می‌, person endings, perfect ‌اند, ezafe ٔ |

## Example

```
Two roads diverged in a yellow wood,      →   Two highways differed in a red timber,
And be one traveler, long I stood         →   And be one visitor, wide I sat

بشنو این نی چون شکایت می‌کند               →   مکن این دف چون شاکی می‌کند
```

## Results

|  | English | Dari |
|---|---|---|
| Content-word tokens | 47 | 60 |
| Found in vocabulary | 100% | 100% |
| Raw top-1 usable as is | 13 (28%) | 10 (17%) |
| Mean cosine similarity of the chosen associates | **0.74** | **0.54** |

Two plots are included: a **PCA map** of each original word and its associate in embedding space, and a **similarity bar chart** for every replacement.

## Key observations

- **The raw nearest neighbour is usually the same word.** It comes back in another form, spelling or with a typo (`traveler → traveller`, `آتش → اتش`, `عشق → وعشق`), because FastText learns from character n-grams.
- **Neighbours are co-hyponyms, not synonyms.** `yellow → red`, `morning → afternoon`, `نی → دف` (one instrument for another).
- **Antonyms are neighbours too.** `black → white` (0.935), `better → worse`, `مرد → زن`. This can flip the poem's logic: "the better claim" becomes "the worse assertion".
- **Polysemy: one vector per word means the wrong sense gets picked.** `wood → timber` (material, not forest), `leaves → left`, `سر` (secret) `→ پا` (it was read as "head"), `باد` ("may it be") `→ وزش` (wind).
- **Persian is harder.** There is no POS tagger or inflection library, words contain zero-width non-joiners, the same word has several spellings, and homographs appear because short vowels aren't written.
- Contextual models such as **BERT** would fix the sense problems that static embeddings can't.

## How to run

```bash
pip install gensim spacy lemminflect pandas numpy matplotlib
python -m spacy download en_core_web_sm
jupyter notebook IWS_3_Poem.ipynb
```

> The first run downloads about 958 MB (English FastText) and streams about 130 MB (Persian vectors). Both are cached for later runs. Run the notebook top to bottom, because Part 2 reuses variable names from Part 1.

## Poems used

Both poems are in the public domain: Robert Frost, *The Road Not Taken* (1916), and Jalal ad-Din Muhammad Balkhi (Rumi), *Masnavi*, Book I, lines 1–10 (13th century).
