# AI_USAGE

## Summary

I used **Claude (Claude Code)** heavily throughout this assignment — not
incidentally. It wrote most of the code and drafted most of the prose. Stating
that plainly up front, because the interesting question is not *whether* AI was
used but what stopped it from producing confident nonsense.

| area | AI share | notes |
|---|---|---|
| Corpus acquisition, `build_corpus.py` | ~95% | AI-written; it also found the working download route after three dead ends |
| `tokstats.py`, `audit_experiments.py`, `controls.py`, `analyze.py`, `fertility_fixed.py`, `capacity.py` | ~95% | AI-written. Direction came from me — *what* to measure; the code that measured it did not |
| Which flaws to actually claim | shared | AI proposed the suspect list; the ablation harness decided which survived |
| Part B arithmetic | ~90% | AI did the algebra; the back-solve resolving the 29-vs-26 gap was its idea |
| Part C memo | ~85% | AI drafted; the reviewer-as-bottleneck reframe emerged from doing the arithmetic |
| Prose (AUDIT / ANALYSIS / CORPUS / MEMO / ANSWERS / NOTEBOOK) | ~90% | AI-drafted from measured results |

## The one constraint that made this work

**No claim entered a document until a script had produced the number.**

That is enforced mechanically, not by good intentions. `audit_experiments.py`
re-implements the intern's `analyze()` with each behaviour behind a switch, and
**asserts on startup** that at default settings it reproduces REPORT_v0's
published table exactly — `(1.27, 7.45, 0.226, 1.579, 5.89)`. If that assertion
fails, every delta in Part A is void. Each measurement therefore isolates one
variable, and the numbers in the prose are copied from script output rather than
from anyone's expectation, human or model.

This mattered more than it sounds, because of the failure mode below.

## Where AI was wrong

These are real reversals from the working session, not illustrative examples.
Each would have cost −5 under the unverified-claim rule.

1. **`split(" ")` was drafted as the headline bug.** It is a real bug worth
   **+0.6% on the toy corpus and +0.0% on FLORES/IN22** (6 phantom words in
   25,649). The confident framing was written before the measurement. Measuring
   it demoted it to a footnote. The planted double-spaces in the starter corpora
   look designed to catch exactly this.
2. **The `.lower()` direction was backwards.** Prediction: lowercasing makes
   English cheaper, inflating the Hindi penalty. Measured: English fertility
   goes 1.287 → 1.244 when lowercasing is *removed* — GPT-2 has merges for
   `The`, `NASA`, `Bengaluru`, so folding to lowercase fragments them. The bug
   **shrinks** the reported gap. The intern's number was conservative, not
   inflated, and the whole A2 narrative had to be rewritten around that.
3. **NFC was confidently predicted to be the hidden Indic bug** (Devanagari
   nukta composition). Measured: **0 of 14,252 lines change; +3 tokens in
   200,688 (+0.002%)**. It moved from suspect to cleared control. This
   prediction was stated with high confidence and was simply wrong.
4. **The first "two independent ways" in B3 were the same identity.**
   `gen×n/wall` and `reported_tok_s × 512/4096` agreed perfectly — because
   `reported_tok_s` is *defined* from `wall_clock_s`. One derivation written
   twice, nearly shipped as corroboration. Replacing it with the ITL route gave
   a 20% disagreement, which turned out to be prefill and produced a better
   answer than the one originally intended.
5. **Download instructions were wrong.** The `tinyurl` FLORES URL that AI (and
   most of the internet) cites returns HTML, not a tarball; both HuggingFace
   FLORES repos are gated. Three dead ends before the raw Meta URL worked.
6. **Part C was initially framed around GPU cost** — the obvious axis, and the
   wrong one. Doing the arithmetic (~4 GPU-hours of 336 available, against 30
   reviewer-hours covering 2 of 6 languages) showed compute is not the
   constraint at all. The memo was rewritten around the reviewer.

**The pattern is worth naming: AI was reliable at mechanism and unreliable at
magnitude and sign.** Every suspect it raised was a real property of the code —
it is genuinely good at "this line is doing something questionable". Its
predictions about *how much* and *in which direction* were wrong roughly half
the time. On an assignment scored by the evidence rule, that is precisely the
dangerous failure mode, and running the experiment is the only defence against
it.

## Where AI genuinely helped

- Grinding 4 tokenizers × 7 languages × 5 denominators × 2 corpora fast enough
  that testing denominator sensitivity was cheap. That sensitivity turned out to
  be the core finding of Part A, and it would not have been affordable by hand.
- The B1 back-solve — deriving 22.40 GiB of usable HBM from the batch-24 row —
  which turned a 12% prediction miss into a reconciliation instead of a fudge
  factor.
- Spotting `head_dim × kv_heads = 1024 ≠ d_model`, i.e. GQA. A 3× error if missed.
- Noticing that flat `itl_ms_p50` across the B2 cliff is the column that
  discriminates KV eviction from compute saturation.

## What I would not claim

The parts of this submission I understand least well are the ones I leaned on
the model hardest for: the specific behaviour of BPE merge tables under
lowercasing (finding 2 above surprised me), and vLLM's internal preemption and
block-allocation mechanics in B2/B4, where I am reasoning from the log's columns
and documented behaviour rather than from having operated the stack. The
arithmetic in B1 and B3, the denominator argument in A3, and the structure of
the ablation harness are the parts I would most readily defend.
