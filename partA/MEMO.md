# A4 — Recommendation memo: Indic tokenizer cost & routing

**To:** Leadership · **Re:** correction to REPORT_v0 §1, before the routing/capacity decision
**Bottom line:** REPORT_v0's direction is right, its magnitude is wrong, and its
recommendation is backwards. Do not budget 6× for Hindi. Change the tokenizer.

## Corrected headline numbers

Cost = tokens needed to express the same content as one English sentence
(FLORES-200 devtest, 1012 parallel sentences; replicated on IN22-Gen within 3%).

| | `gpt2` (v0's tokenizer) | `sarvam-1` (Indic-aware) |
|---|---|---|
| Hindi | **7.42×** English | **1.13×** |
| Kannada | **13.58×** | 1.21× |
| Tamil | **15.54×** | 1.16× |
| Telugu | 12.97× | 1.15× |
| Bengali / Marathi | 9.61× / 7.86× | 1.12× / 1.07× |

Three corrections to what is currently in the deck:

1. **"5.89× for Hindi" is wrong in both directions.** Computed correctly on a
   real corpus it is 7.42× on GPT-2 — worse than reported. Computed with a
   tokenizer we would actually deploy, it is 1.13×.
2. **"6× for Indic" hides a 2× spread.** Tamil and Kannada are ~2× worse than
   Hindi on GPT-2. v0 measured Hindi, the *best-case* Indic language, and
   generalised to all of it.
3. **"A property of the script, not the tokenizer" is false.** Same corpus,
   same metric, swap only the tokenizer: 7.42× → 1.13×. ~85% of the penalty
   is GPT-2's training coverage. It is a fixable defect, not a fact of Devanagari.

## Recommendation

**Do not create an Indic routing tier, and do not budget a 6× Indic multiplier.**
Both price in a defect instead of removing it.

1. **Make tokenizer coverage the selection criterion** for the model serving
   Indic traffic. On a like-for-like decoder comparison the Indic-aware
   tokenizer costs 1.13–1.21× English across all six languages — a spread of
   7%, small enough that a single capacity number covers all of them.
2. **If we must stay on the current model near-term,** budget from the
   per-language table, not from Hindi: provision Dravidian traffic at ~13–16×,
   not 6×. Under-provisioning Tamil by 2.5× is the concrete failure mode of
   shipping the deck as written.
3. **Re-run this before signing anything**, on replayed production traffic
   rather than FLORES. `partA/fertility_fixed.py` takes a corpus and does it.

## The biggest caveat

**My corpus is not our traffic.** FLORES and IN22 are edited, well-formed,
native-script written prose. Real Indic assistant traffic is short, code-mixed,
and frequently *romanised* — and romanised Hindi tokenizes roughly like English,
showing no Indic penalty at all. If a large share of our Indic traffic is
Latin-script, every multiplier above is an over-estimate and the true cost gap
is smaller than I am claiming. I could not size this from a parallel corpus.
Both corpora are also translated from English, which plausibly makes the Indic
side easier to tokenize than native text — biasing the same direction.

Secondary: these are input-side measurements, and serving cost is dominated by
output tokens; and a 197k-vocab tokenizer buys fewer tokens at the price of a
bigger embedding/LM-head, which I have not priced.

## The one metric to monitor in production

**Median output tokens per completed request, segmented by request language,
expressed as a ratio to English — tracked weekly.**

- It is measured on real traffic, so it closes the corpus caveat above by
  construction.
- It is on the **output** side, which is where the cost and latency actually
  are, closing my second caveat.
- It needs no eval set, no annotation and no parallel data — just the token
  counters the serving stack already emits.

**Expected value if this analysis is right:** ~1.1–1.3× on an Indic-aware
tokenizer; 7–16× on GPT-2-class, ordered Hindi < Marathi < Bengali < Telugu <
Kannada < Tamil.

**Alarm:** the ratio for any language sitting outside its predicted band by
more than ~30%, or the Tamil/Hindi ratio collapsing toward 1.0. Either means my
corpus does not represent our traffic — most likely because it is romanised or
code-mixed — and this analysis should be redone on replayed logs before any
capacity money is committed.
