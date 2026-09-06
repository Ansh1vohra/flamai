# AI_USAGE

> **Read this first, and edit it before you submit.** This file was drafted at
> the end of the session by the same AI that did the work described in it, so
> the parts about *my own* understanding are the parts I cannot write for you.
> Sections 1–3 are a factual record of what the tool did and where it was wrong;
> those are accurate. Section 4 is a checklist you have to be able to satisfy
> yourself — go through it, and cut anything from this repo you cannot defend.
> The assignment is explicit that this is not a trap, and that an honest
> "the AI wrote most of this" is a better answer than a flattering one.

## 1. What AI did

I used **Claude (Claude Code)** throughout, in one long session. It was heavy
use, not incidental:

| area | AI share | what that means |
|---|---|---|
| Corpus acquisition + `build_corpus.py` | ~95% | It wrote it; it also found the ungated download route after three gated/broken ones |
| `tokstats.py`, `audit_experiments.py`, `controls.py`, `analyze.py`, `fertility_fixed.py`, `capacity.py` | ~95% | AI-written. I directed *what to measure*; it wrote the code that measured it |
| Choice of which flaws to claim | shared | AI proposed a suspect list; the ablation harness decided which survived |
| Part B arithmetic | ~90% | AI did the algebra; the back-solve that resolved the 29-vs-26 gap was its idea |
| Part C memo | ~85% | AI drafted; the reviewer-as-bottleneck reframe came out of it doing the arithmetic |
| Prose (AUDIT/ANALYSIS/CORPUS/MEMO/ANSWERS/NOTEBOOK) | ~90% | AI-drafted from the measured results |

Method that mattered more than the split: **no claim went into a document until
a script had produced the number.** The ablation harness asserts on startup that
it reproduces REPORT_v0's published table exactly, so every delta is attributable
to one switch. That constraint is what stopped the plausible-sounding wrong
claims below from shipping.

## 2. Where AI was wrong, and measurement caught it

These are real reversals from this session, not illustrative examples. Each one
would have cost −5 under the unverified-claim rule.

1. **`split(" ")` was drafted as the headline bug.** It is real, and it is worth
   **+0.6% on the toy corpus and +0.0% on FLORES/IN22** (6 phantom words in
   25,649). The confident framing was written before the measurement; the
   measurement demoted it to a footnote. The planted double-spaces in the
   starter corpora look designed to catch exactly this.
2. **The `.lower()` direction was backwards.** The stated prediction was that
   lowercasing makes English cheaper and so *inflates* the Hindi penalty.
   Measured, English fertility goes 1.287 → 1.244 when lowercasing is removed —
   lowercasing makes English *more expensive* (GPT-2 has merges for `The`,
   `NASA`, `Bengaluru`), so the bug **shrinks** the reported gap. The whole
   A2 narrative had to be rewritten; the intern's number was conservative, not
   inflated.
3. **NFC was confidently predicted to be the hidden Indic bug** (Devanagari
   nukta composition). Measured: **0 of 14,252 lines change, +3 tokens in
   200,688 (+0.002%)**. It moved from "suspect" to "cleared control". This one
   was stated with high confidence and was simply wrong.
4. **The first "two independent ways" in B3 were the same identity.**
   `gen×n/wall` and `reported_tok_s × 512/4096` agreed perfectly — because
   `reported_tok_s` is *defined* from `wall_clock_s`. That is one derivation
   written twice, and it was nearly shipped as corroboration. Replaced with the
   ITL-based route, which touches neither column and disagreed by 20% — a
   disagreement that turned out to be prefill and produced a better answer.
5. **The initial FLORES download instructions were wrong** — the `tinyurl` URL
   that AI (and most of the internet) cites returns HTML, and both HuggingFace
   FLORES repos are gated. Three dead ends before the working raw URL.
6. **Part C was initially framed around GPU cost**, the obvious axis. Doing the
   arithmetic (~4 GPU-hours of 336 available vs 30 reviewer-hours covering 2 of
   6 languages) showed compute is not the constraint at all, and the memo was
   rewritten around the reviewer.

Pattern worth naming: **AI was reliable at mechanism and unreliable at
magnitude and sign.** Every suspect it raised was a real property of the code.
Its predictions about *how much* and *in which direction* were wrong roughly
half the time. In an assignment scored on the evidence rule, that is the
dangerous failure mode, and the only defence is running the experiment.

## 3. Where AI genuinely helped

- Grinding out 4 tokenizers × 7 languages × 5 denominators × 2 corpora quickly
  enough that checking the denominator sensitivity was cheap. That sensitivity
  is the core finding, and it would not have been affordable by hand.
- The B1 back-solve — deriving 22.40 GiB of usable HBM from the batch-24 row —
  which turned a 12% miss into a reconciliation instead of a fudge factor.
- Spotting that `head_dim × kv_heads = 1024 ≠ d_model`, i.e. GQA, which is a 3×
  error if missed.
- Noticing that flat `itl_ms_p50` across the B2 cliff is the column that
  discriminates KV eviction from compute saturation.

## 4. What you need to be able to do before the defense

The repo is only as good as what you can re-derive live. Work through these; if
one of them doesn't hold, cut the claim rather than defend it thin.

- [ ] Re-derive **114,688 bytes/token** on a whiteboard, and say why it's 8 KV
      heads and not 24.
- [ ] Re-derive **26 sequences**, and explain the 24 GiB vs 22.4 GiB gap.
- [ ] Explain why `reported_tok_s` inflates with prompt length, and compute the
      batch-24 goodput (**200.9 tok/s**) two ways — including why the ITL route
      gives 249.8 and what the 20% gap is.
- [ ] Explain, without notes, why **tokens-per-parallel-sentence** is the right
      denominator and tokens-per-word is not. This is the highest-value question
      in the whole assignment and it will be asked.
- [ ] Say why `split(" ")` is a real bug you are *not* leading with, and give
      the number.
- [ ] Say why `random.seed`, NFC and `add_special_tokens=False` are **not**
      bugs, with the measurement for each.
- [ ] Run `audit_experiments.py`, `controls.py`, `analyze.py`, `capacity.py`
      cold, on a fresh clone, and know how long each takes.
- [ ] Be ready for "run your script on this input I'm about to paste" —
      `fertility_fixed.py` takes arbitrary `LANG=PATH` pairs and any
      `hf:<repo_id>` tokenizer; try it once on something you didn't prepare.
- [ ] Know the four open threads at the end of NOTEBOOK.md. Being asked "what
      would change your mind?" is likely, and those are the answers.
