# A3 — Corrected cross-language analysis

**Reproduce:** `python partA/analyze.py` → `results/a3_analysis.md`, `results/a3_analysis.csv`
Full tables (4 tokenizers × 7 languages × 5 denominators × 2 corpora) are in
`results/a3_analysis.md`. This file is the reasoning.

Setup: 4 tokenizers, 5 denominators, micro-averaged, no lowercasing, NFC,
`add_special_tokens=False`. Numbers below are FLORES devtest (1012 parallel
sentences); IN22-Gen replicates all of them within ~3% ([CORPUS.md](CORPUS.md)).

| tokenizer | kind | vocab |
|---|---|---|
| `gpt2` | byte-level BPE, English-trained — what REPORT_v0 used | 50k |
| `cl100k_base` | byte-level BPE, multilingual-ish, no Indic specialisation | 100k |
| `sarvamai/sarvam-1` | Indic-first **decoder LLM** tokenizer, 10 Indian languages | 68k |
| `google/muril-base-cased` | Indic-specialised **encoder** (WordPiece), 17 Indian languages | 197k |

## 1. The denominator, not the tokenizer, is what REPORT_v0 actually reported

GPT-2, Hindi vs English, from **one** set of token counts (200,688 Hindi /
27,044 English tokens over the same 1012 sentences):

| denominator | penalty | what it holds constant |
|---|---|---|
| UTF-8 byte | **2.90×** | storage volume |
| whitespace word | **6.34×** | orthographic words — *not* an amount of meaning |
| **parallel sentence** | **7.42×** | **content** |
| grapheme cluster | **11.39×** | user-perceived characters |

A 3.9× spread from a single tokenizer run. Any of these could headline a deck.

The reason they diverge is that the denominators are not language-neutral:

| lang | words/sent | graphemes/sent | bytes/sent | graphemes/word | bytes/grapheme |
|---|---|---|---|---|---|
| eng | 21.64 | 130.4 | 130.5 | 6.03 | 1.00 |
| hin | 25.34 | 84.9 | 333.4 | 3.35 | 3.93 |
| ben | 19.27 | 80.7 | 344.6 | 4.19 | 4.27 |
| kan | 15.91 | 89.3 | 370.9 | 5.61 | 4.15 |
| tam | 16.58 | 98.5 | 416.6 | 5.94 | 4.23 |
| tel | 16.74 | 75.7 | 349.5 | 4.52 | 4.62 |

Identical content, 21,901 English words but 16,100 Kannada words (0.74×) and
25,649 Hindi words (1.17×). "Per word" divides by a different quantity of
meaning in every row — and biases in *opposite directions* for Hindi (flattered)
and Kannada (penalised). "Per grapheme" and "per byte" are worse still: Indic
scripts pack ~4 bytes per grapheme against English's 1.00.

## 2. Which single number should drive routing and cost

**Tokens per parallel sentence, relative to English — the `cost vs eng` column.**

The reasoning, not the assertion:

1. The decision is *what will it cost, and where should traffic go*. The unit
   we are billed in is the token; the unit the user cares about is one
   answered request. So the ratio that matters is **tokens per unit of
   delivered content**.
2. A denominator's job is to hold the confound constant. The confound is "how
   much is being said". Words, graphemes and bytes are all *orthographic*
   properties of how a language writes things down — they vary 0.74×–1.17×
   across languages for identical content, and they vary for reasons that have
   nothing to do with the tokenizer. Only "one parallel sentence" is defined as
   *the same meaning in every language*.
3. It is also the only column that composes into money without a second
   assumption. `cost_vs_eng = 7.42` means: route a Hindi request that says what
   an English request says, and you pay 7.42× the tokens. Prefill FLOPs, KV
   cache, decode steps and context-window pressure all scale with it directly.
   No other row here converts to a budget line without an extra conversion.
4. Byte-fertility is the useful *diagnostic* second number — it isolates
   "how well does the BPE handle this script" from "how verbose is this
   language" — but it is not the cost number, because nobody is billed per byte.

**Concretely: the number I would put on the slide is the `cost vs eng` column,
per language, per candidate tokenizer.**

The honest limit of that choice, stated up front: it is only defined on a
parallel corpus, and a FLORES sentence is not a chat turn. It is the right
*shape* of number; the right *value* comes from replaying real traffic
([MEMO.md](MEMO.md) monitoring metric).

## 3. Corrected headline table — cost vs English, per parallel sentence

| lang | `gpt2` (v0's) | `cl100k` | `muril` | `sarvam-1` |
|---|---|---|---|---|
| eng | 1.00× | 1.00× | 1.00× | 1.00× |
| hin | **7.42×** | 4.77× | 1.16× | **1.13×** |
| ben | 9.61× | 5.88× | 1.00× | 1.12× |
| mar | 7.86× | 5.05× | 1.06× | 1.07× |
| kan | **13.58×** | 8.86× | 1.07× | **1.21×** |
| tam | **15.54×** | 7.64× | 1.06× | **1.16×** |
| tel | 12.97× | 8.29× | 1.20× | 1.15× |

Three findings, each of which contradicts REPORT_v0:

**(a) The penalty is a tokenizer property, not a script property.** REPORT_v0
finding 3 says "any tokenizer will struggle… a property of the script". Holding
corpus and denominator fixed and changing only the tokenizer moves Hindi from
7.42× to 1.13× — **~85% of the penalty is GPT-2, not Devanagari.** `sarvam-1`
achieves it with a *smaller* vocab than `cl100k` (68k vs 100k), so this is about
training-data coverage, not vocabulary size.

**(b) v0's own headline was understated, not overstated.** On the metric v0
chose, correctly computed, Hindi is 6.34× per word (not 5.89×) and 7.42× per
sentence. Fixing the code bugs makes GPT-2 look *worse*. The report is wrong
about the recommendation, not about the direction of its own number.

**(c) "All Indic traffic" is not one bucket — and this is the finding v0 could
not have made.** v0 measured only Hindi and generalised. Tamil is **2.1× worse
than Hindi** on GPT-2 (15.54× vs 7.42×) and Kannada 1.8× worse. Hindi is the
*best-case* Indic language, because Devanagari has the most representation in
web-scraped BPE training data. Any capacity plan built on the Hindi number
under-provisions Dravidian traffic by roughly 2×.

Note (c) largely disappears under an Indic tokenizer (1.13×–1.21×, a spread of
7%), which is a second, independent way of seeing (a): the cross-language
variance *is* the tokenizer's coverage gap.

## 4. Caveats on this table

- `muril` is an **encoder** (BERT/WordPiece). Its numbers are valid as an
  "Indic-aware tokenizers can do this" existence proof, but you cannot serve
  generation on it. `sarvam-1` is the decoder-shaped comparator and it is the
  one I would quote in a routing decision.
- `muril` reaches 1.00× on Bengali and 1.16× on Hindi, i.e. it is *slightly
  better than English*. That is not free: 197k vocab means a larger embedding
  and LM-head matrix and more softmax compute per step. Token count is one term
  in the cost function, not the whole of it. I have not measured that trade-off.
- These are **input-side** measurements. Serving cost is dominated by output
  tokens; I assume the ratio transfers to generations and have not verified it.
- All four tokenizers are open. Switching a *tokenizer* means retraining or at
  minimum re-embedding the *model* — a real cost this analysis does not price.
  The finding is "the tokenizer is the lever", not "flip a config flag".
