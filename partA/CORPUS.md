# A1 — The eval corpus

**Build it:** `python partA/build_corpus.py` (downloads ~26 MB, ~1 min)
**Verify it:** `python partA/build_corpus.py --check` (prints per-file SHA-256)

## What I assembled

Two independent parallel corpora, not one — because a single corpus cannot tell
me whether a finding is about tokenizers or about the domain I happened to pick.

| | **FLORES-200 devtest** | **IN22-Gen** |
|---|---|---|
| source | NLLB / Meta, `dl.fbaipublicfiles.com` | AI4Bharat, via the ungated `mteb/IN22-Gen` mirror |
| sentences | **1012**, line-parallel | **1024**, line-parallel |
| domain | Wikipedia-derived news/encyclopedic prose, Western-skewed subject matter | Indian-context sources across 13 domains (news, legal, culture, government, education …) |
| direction | English source → professionally translated | English source → professionally translated |
| role here | primary (it is the standard the assignment names) | independent domain replication |

**Languages (7):** `eng`, `hin`, `ben`, `mar`, and three Dravidian —
`kan`, `tam`, `tel`. The brief requires ≥4 including English, Hindi and two
Dravidian; I carry seven because Part C names exactly these six Indic
languages, and because Kannada/Tamil turn out to behave very differently from
Hindi (see [ANALYSIS.md](ANALYSIS.md)) — a four-language set would have hidden
the most decision-relevant finding.

**Total measured:** 7 languages × 2036 sentences = **14,252 lines**, roughly
1.4 M characters. Every result in Part A is computed over all of it.

## Preprocessing — deliberately almost none

`build_corpus.py::norm()` does exactly two things:

1. `str.strip()` — remove the trailing newline and surrounding whitespace.
2. `unicodedata.normalize("NFC", …)` — so that the byte and grapheme
   denominators are well-defined. Measured to be a no-op on both corpora
   (0 of 14,252 lines change; see [AUDIT.md](AUDIT.md) C6) — it is a guarantee,
   not a transformation.

And explicitly **not**: no lowercasing (that is bug F2), no punctuation
stripping, no de-duplication, no length filtering, no romanisation. The premise
is that fertility should be measured on text shaped like the text you will
actually serve, and every cleaning step is a silent thumb on the scale.

A **parallel-alignment guard** drops any sentence index where *any* of the 7
languages is empty, so line *i* means the same thing in all 7 files — this is
what makes the per-parallel-sentence denominator legitimate. It dropped 0 lines
from both corpora, which is itself the check that the alignment is intact.

## What this corpus **cannot** tell me

Five things, and the first two are the ones that could actually reverse a
decision:

**It is not our traffic.** Both corpora are edited, well-formed, full-sentence
written prose with correct orthography. An assistant's real Indic input is
short, code-mixed, romanised as often as not (`kal office jaana hai`), full of
typos, emoji, product names and English technical nouns. Romanised Hindi
tokenizes roughly like English and would *not* show an Indic penalty at all, so
if a meaningful share of our traffic is Latin-script Indic, every multiplier
here is an over-estimate of our real cost. I cannot size that error from this
corpus. **This is the caveat I would spend the next day closing**, and the
production metric in [MEMO.md](MEMO.md) is designed to catch it.

**It is translationese.** Both corpora were translated *from* English, so the
targets track English sentence structure more closely than native writing does.
Plausibly this makes the Indic side *easier* to tokenize than genuine native
text, again biasing my multipliers optimistically. Direction argued, magnitude
unmeasured — I flag it rather than claim it.

**It measures the prompt side, not the reply side.** Fertility here is computed
on input text. Serving cost and latency are dominated by *output* tokens, and I
have not measured the model's Indic generations. I assume the ratio transfers;
that assumption is untested.

**It is one register.** No conversational, spoken-transcript or code-mixed
register. Since Part C is explicitly about making replies *casual*, that gap
sits directly under the thing the product team wants to ship.

**Sample size is adequate for the effect I am claiming, and only that.** ~1000
sentences per language pins each aggregate token ratio to well within a percent
— fine, because the effects that matter here are 2× to 15×, not 5%. It is *not*
enough to rank two similar tokenizers that differ by a few percent, and I do not
do so anywhere. Where two of my numbers are within ~5%, I treat them as tied.

**One real cross-corpus check I did run.** FLORES and IN22 differ in domain,
provenance and era, so agreement between them is evidence, not decoration. GPT-2
cost-vs-English per parallel sentence:

| lang | FLORES | IN22 | Δ |
|---|---|---|---|
| hin | 7.42× | 7.27× | −2.0% |
| kan | 13.58× | 13.63× | +0.4% |
| tam | 15.54× | 15.16× | −2.4% |
| tel | 12.97× | 12.55× | −3.2% |

Every conclusion in [ANALYSIS.md](ANALYSIS.md) survives the swap within ~3%. So
the findings are robust to *domain within edited written prose* — which is
exactly the scope of the guarantee, and says nothing about the code-mixed
conversational traffic caveated above.
