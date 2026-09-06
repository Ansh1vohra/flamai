# A2 — Audit of `fertility.py` and of the metric it computes

**Reproduce everything here with:**

```bash
python partA/audit_experiments.py     # -> results/a2_audit.txt  (F1–F4, F6)
python partA/controls.py              # -> the three cleared suspects (C5–C7)
```

`audit_experiments.py` re-implements v0's `analyze()` with each behaviour behind
a switch, and **asserts on startup** that at default settings it reproduces
REPORT_v0's published table exactly — `(1.27, 7.45, 0.226, 1.579, 5.89)`. Every
delta below is therefore attributable to the single switch that was flipped.

Baseline for all deltas, GPT-2, as-shipped:

| corpus | eng fert | hin fert | **fert ratio** | eng tok/char | hin tok/char | tok/char ratio |
|---|---|---|---|---|---|---|
| starter sample (10 sent) | 1.265 | 7.448 | **5.89×** | 0.226 | 1.579 | 7.00× |
| FLORES devtest (1012) | 1.287 | 7.865 | **6.11×** | 0.214 | 1.529 | 7.14× |
| IN22-Gen (1024) | 1.377 | 8.332 | **6.05×** | 0.221 | 1.539 | 6.98× |

---

## Summary table

| # | Claim | Kind | Measured effect on the headline 5.89× | Verdict |
|---|---|---|---|---|
| **F1** | `line.split(" ")` counts empty strings as words | code bug | +0.6% (sample), **+0.0%** (FLORES/IN22) | real, **immaterial here** |
| **F2** | `line.lower()` is a one-sided edit | code bug | **+3.4%** (5.89→6.06 sample, 6.11→6.32 FLORES) | real, material, **and it flatters Hindi** |
| **F3** | mean-of-ratios instead of sum/sum | method bug | +0.3% / +0.2% / −0.2% | real, immaterial here |
| **F4** | `chars = len(line)` is not "characters" | **conceptual** | tok/char ratio moves **7.14× → 10.93× → 2.78×** from identical tokens | real, **large** |
| **F5** | tok/word and tok/char hold nothing constant across languages | **conceptual, the main one** | the "Indic penalty" is 2.90×, 6.34×, 7.42× or 11.39× depending only on the divisor | real, **decisive** |
| **F6** | REPORT_v0: "the two metrics agree, so the result is robust" | reporting bug | they share a numerator; they are not independent | real |
| **F7** | REPORT_v0: "property of the script, not the tokenizer" | reporting bug | swapping the tokenizer moves 7.42× → **1.13×** | real, **falsified** |
| C5 | `random.seed(1337)` | *suspect, cleared* | **exactly 0** (byte-identical output) | **not a bug** |
| C6 | `unicodedata.normalize("NFC", ...)` | *suspect, cleared* | **0 lines changed in 14,252** | **not a bug** |
| C7 | `add_special_tokens=False` | *suspect, cleared* | correct for a ratio; would shrink it toward 1.0 | **not a bug** |

---

## F1 — `line.split(" ")` fabricates words (real, but small; I nearly over-claimed it)

`analyze()` does `words = line.split(" ")`. Python's `str.split(" ")` splits on
each *single literal space*, so a run of two spaces yields an empty string that
is then counted as a word. `str.split()` (no argument) splits on runs of any
whitespace and never yields empties.

Both starter corpora contain a planted double space (`eng_sample.txt:7`
"the books  in", `hin_sample.txt:10` "किताबें  अलमारी").

```
    -> phantom empty 'words' created by split(' '): eng=1 hin=1 (of 79/62 counted)
F1  fix: .split() on whitespace   1.283  7.598  5.92  | fert_ratio 5.89 -> 5.92 (+0.6%)
```

Inflating the denominator *deflates* fertility, so v0 under-reports both
languages. On the published sample the ratio moves 5.89× → 5.92×.

**Honest magnitude.** On my real corpora the effect is *nil*: FLORES has 6
phantom words out of 25,649 Hindi words and 0 of 21,901 English; IN22 has none.
The ratio moves by less than 0.05%.

> I originally wrote this up as a headline bug. Measuring it demoted it. It is a
> genuine defect and a latent landmine — it scales with whatever whitespace
> noise a future corpus carries, and it fails silently — but **it is not why
> REPORT_v0 is wrong**, and I would not have known that without running it.

## F2 — `line.lower()` is a one-sided edit (real, material, and it works *against* the report's own thesis)

```python
line = line.lower()   # "lowercase so casing doesn't add noise to the comparison"
```

The comment states the intent and the intent is wrong: `.lower()` is a no-op for
Devanagari, Kannada, Tamil, Telugu and Bengali, which are unicameral. So it
edits one arm of a two-arm comparison and leaves the other untouched.

```
    -> lines actually changed by .lower(): eng=1012 hin=38   [FLORES]
F2  fix: no .lower()   1.244  7.865  6.32  | fert_ratio 6.11 -> 6.32 (+3.4%)
```

(38 Hindi lines change because FLORES Hindi carries Latin-script proper nouns.)

**Direction and magnitude — this is the interesting part.** Lowercasing makes
English *worse*, not better: GPT-2's BPE has dedicated merges for capitalised
forms, so `NASA`, `ISRO`, `Bengaluru`, `The` fragment when folded to lowercase.
English fertility falls 1.287 → 1.244 when the bug is removed. The bug therefore
**inflates the English baseline and shrinks the reported gap**: the honest
number is 6.32×, not 6.11×.

So this bug was *helping* REPORT_v0's headline look conservative. Fixing v0's
code makes Hindi look worse, not better. The report is wrong for a different
reason — F5.

## F3 — mean-of-ratios instead of sum-over-sum (real, immaterial at this corpus composition)

`analyze()` accumulates `len(tokens)/len(words)` per line and averages. That
weights a 3-word line the same as a 40-word line, and E[X/Y] ≠ E[X]/E[Y].
For a **cost** question the correct aggregation is unambiguous: you are billed
for the corpus, so it is Σtokens / Σunits.

```
F3  fix: micro-average (sum/sum)   1.278  7.825  6.12  | 6.11 -> 6.12 (+0.2%)
```

**Measured effect: +0.2% (FLORES), +0.3% (sample), −0.2% (IN22).** I am claiming
this as a methodology defect with a measured magnitude of ~0.2%, not as a
cause of the report's error. It is small here only because FLORES line lengths
are homogeneous; it would bite on real traffic, where a chat corpus mixes
3-token and 3000-token requests.

## F4 — `chars = len(line)` is not a count of characters (conceptual, large)

`len(line)` counts Python code points. For Latin text that coincides with what a
reader would call a character. For Indic scripts it does not: a single
user-perceived character (an akshara) is a base consonant plus matras, virama
and nukta — several code points. And for a *cost* question the physically
meaningful unit is UTF-8 bytes, where Devanagari costs 3 bytes/code point.

There are three defensible things "tok/char" could mean, and they give three
different answers from **identical token counts**:

| denominator (FLORES, GPT-2, Hindi vs English) | hin tok/unit | eng tok/unit | ratio |
|---|---|---|---|
| code points (`len(line)`, what v0 does) | 1.529 | 0.214 | **7.14×** |
| Unicode extended grapheme clusters (`\X`) | 2.341 | 0.214 | **10.93×** |
| UTF-8 bytes | 0.595 | 0.214 | **2.78×** |

A **3.9× spread** with no change to the tokenizer, the corpus, or a single
token. v0 picked one silently and reported it to three decimal places.

## F5 — The conceptual problem: neither denominator holds anything constant

This is the one that matters, and it is not a coding error — the code computes
tokens-per-word correctly. Tokens-per-word is the wrong quantity.

A denominator in a cross-language ratio exists to hold something constant so the
numerator is comparable. "Word" and "character" hold nothing constant, because
languages package meaning differently:

FLORES, the same 1012 sentences in every language:

| lang | words/sentence | graphemes/word | words vs eng |
|---|---|---|---|
| eng | 21.64 | 6.03 | 1.00× |
| hin | 25.34 | 3.35 | 1.17× |
| ben | 19.27 | 4.19 | 0.89× |
| kan | 15.91 | 5.61 | **0.74×** |
| tam | 16.58 | 5.94 | 0.76× |
| tel | 16.74 | 4.52 | 0.77× |

The identical content is 21,901 English words but 16,100 Kannada words —
Kannada is agglutinative, so one orthographic word carries what English spreads
over several. Dividing by "words" therefore divides by a *different amount of
meaning* in every row, and it does so in opposite directions: it flatters Hindi
(more words per sentence than English) and penalises Kannada (fewer).

The full consequence, GPT-2, Hindi vs English, from one set of token counts:

| denominator | penalty |
|---|---|
| per UTF-8 byte | 2.90× |
| per word | 6.34× |
| **per parallel sentence** | **7.42×** |
| per grapheme cluster | 11.39× |

**Any of these is a defensible "fertility". Only one of them answers the
question leadership is asking.** You are billed per token for delivering a
user's *intent*; the thing that must be held constant is content, not
orthography. On a parallel corpus that is tokens-per-parallel-sentence — worked
through in [ANALYSIS.md](ANALYSIS.md).

## F6 — "The tok/char column agrees … which confirms the per-word number"

REPORT_v0 finding 2 treats tok/char as independent corroboration. It is not:
both columns have the *same numerator*. Identically,

```
tok/char ratio  =  tok/word ratio  ×  (words-per-codepoint ratio)
     7.4653     =      6.3379      ×        1.1779              [exact, micro-averaged]
```

(That is an algebraic identity, not an empirical finding — it holds to all
printed digits; verify it with the snippet in NOTEBOOK.md, entry 6. Under v0's
macro aggregation it holds approximately: 6.11 × 1.169 ≈ 7.14.)

The second metric adds exactly one new fact — the chars-per-word ratio between
the two languages — and nothing about the tokenizer. Two metrics sharing a
numerator agreeing is not robustness. And per F4 they do not even agree: change
"char" from code point to grapheme and the "confirmation" reads 10.93×.

This is what licenses the report's closing line, **"No further measurement
needed"** — which is the single most expensive sentence in the document.

## F7 — "Root cause: … a property of the script, not the tokenizer"

REPORT_v0 finding 3 is a causal claim, and it is falsifiable by holding the
corpus fixed and swapping only the tokenizer. FLORES, Hindi vs English, cost per
parallel sentence:

| tokenizer | vocab | hin cost vs eng |
|---|---|---|
| `gpt2` (what the intern used) | 50k | **7.42×** |
| `cl100k_base` | 100k | 4.77× |
| `google/muril-base-cased` | 197k | **1.16×** |
| `sarvamai/sarvam-1` | 68k | **1.13×** |

Nothing about Devanagari changed between those rows. **The swap removes ~85% of
the measured multiplier, so the penalty is largely tokenizer-dependent, not a
property of the script** — which is the claim REPORT_v0 makes. Which *tokenizer*
property is responsible is not isolated here; vocabulary size alone is ruled out
(`sarvam-1` reaches 1.13× with 68k against `cl100k`'s 100k), leaving training
corpus, merge rules and pre-tokenisation uncontrolled.

The recommendation that follows from this finding — "budget 6× serving cost for
Hindi" — therefore prices in a defect that costs nothing to remove.

---

## Cleared suspects — things that look wrong and are not

Run `python partA/controls.py`. All three were on my first suspect list.

### C5 — `random.seed(1337)  # reproducibility`

Reads as though the script samples the corpus and reports one draw. It does not:
`random` is imported, seeded, and never used again. `controls.py` greps the body
for other uses (none), then deletes both lines and re-runs the shipped script:

```
  with seed   : hin is 5.89x the fertility of eng (worse tokenization)
  seed removed: hin is 5.89x the fertility of eng (worse tokenization)
  outputs identical: True   -> effect on reported numbers: 0.000 (exactly)
```

Dead code and a misleading comment. **Not a bug.**

### C6 — `unicodedata.normalize("NFC", line)`

My first instinct was that this was the hidden Indic bug: NFC composition merges
code points, which would move any per-code-point denominator, and Devanagari has
composable nukta forms (क़ ज़ फ़). Measured across all 14,252 lines of both
corpora in all 7 languages: **zero lines change, zero token delta.** Both
corpora already ship NFC.

I then built the adversarial case — feed the pipeline NFD text — and the
normalisation absorbs it: identical output. Even the raw NFD/NFC token
difference is +3 tokens in 200,688 (+0.002%).

So the call is (a) measurably inert on this data, (b) the correct defensive
choice, since the byte and grapheme denominators are only well-defined for a
known normalisation form. **Not a bug.** *(Where I was wrong: I expected nukta
composition to matter measurably. It does not — the effect is ~0.002%, not the
percent-level I predicted.)*

### C7 — `add_special_tokens=False` on the HF path

Looks like an undercount, because production really does prepend BOS and you
really do pay for it. But for a cross-language *ratio* it is correct: specials
add a constant per sequence, which dilutes every ratio toward 1.0.

```
  add_special_tokens=False  eng_fert=1.259 hin_fert=1.245 ratio=0.989
  add_special_tokens=True   eng_fert=1.352 hin_fert=1.324 ratio=0.979
```

+2 tokens/sentence to both arms, ratio moves 0.989 → 0.979. Excluding them is
right for the metric. **Not a bug** — but the caveat is carried into
[MEMO.md](MEMO.md): an absolute *billing* estimate must add them back, along
with the chat template.

---

## Two defects I found but am **not** claiming as headline flaws

Recorded for completeness; neither distorts the published numbers.

1. **`base = langs[0]`** — the comparison baseline is whichever `--corpus` was
   typed first, and the printed sentence ("*worse*"/"*better*") flips with
   argument order. Real fragility, zero effect on the table as run. Fixed in
   `fertility_fixed.py` via an explicit `--baseline`.
2. **No division guard** — `len(words)` and `chars` can be 0 for a line that is
   pure whitespace. `read_lines` happens to filter those, so it never fires on
   this data. Latent, not active.
