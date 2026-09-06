# NOTEBOOK

Chronological log: hypothesis → experiment → result → revision. Written at the
end of each work block, not keystroke-by-keystroke, so the prose is tidier than
the process was — but the dead ends and reversals below are the real ones, in
the order they happened. Everything marked ✗ is something I believed and then
had to unbelieve.

---

## Block 1 — Read the inputs, form suspicions

Read the assignment, `fertility.py`, `REPORT_v0.md`, `model_spec.md`,
`bench_log.csv`. First reproduction, before touching anything:

```bash
python starter_kit/fertility.py \
  --corpus eng=corpus_sample/eng_sample.txt \
  --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
# eng 1.27 / hin 7.45 / 0.226 / 1.579 -> "hin is 5.89x"
```

Exact match to the published table. Good — the report is at least honest about
what the script produced.

Suspect list, written before any measurement (so I can be graded on how much of
it survived):

1. `line.split(" ")` — literal-space split; double spaces make phantom words.
   Both sample files contain a planted double space. **Predicted: headline bug.**
2. `line.lower()` — unicameral scripts are unaffected; one-sided edit.
   **Predicted: deflates the gap, because lowercasing should make English
   cheaper.**
3. `random.seed(1337)` — is there sampling somewhere I haven't found?
4. `unicodedata.normalize("NFC")` — **Predicted: this is the hidden Indic bug.**
   Devanagari has composable nukta forms (क़ ज़ फ़); normalisation moves code
   point counts, which moves the tok/char denominator.
5. Mean-of-ratios aggregation instead of sum/sum.
6. `chars = len(line)` counts code points, which is not what "character" means
   in an Indic script.
7. Report finding 2 ("the tok/char column agrees, which confirms") — the two
   columns share a numerator. That cannot be independent confirmation.
8. Report finding 3 ("property of the script, not the tokenizer") — a causal
   claim, and trivially falsifiable by swapping the tokenizer.

Note to self at this point: the assignment says one suspicious thing is
harmless, so at least one of 1–6 is a trap. I guessed the trap was #3.

**Result: I was right about the trap being #3, and wrong about #1, #2 and #4 —
all three in ways I could only find by measuring.**

---

## Block 2 — Corpus acquisition (the messy part)

Wanted FLORES-200 as A1's primary corpus.

- ✗ `https://tinyurl.com/flores200dataset` — the URL everyone cites. Returned a
  10 KB HTML page, not a tarball. `tar: not in gzip format`.
- ✗ `openlanguagedata/flores_plus` on HF — **401 GatedRepoError.**
- ✗ `facebook/flores` on HF — also **401 GatedRepoError.** Both need an
  accepted licence + token, which I did not want as a reproduction dependency
  for the defense (the grader has to be able to run this).
- ✓ Searched for ungated mirrors, found `mteb/IN22-Gen` — AI4Bharat's 1024
  parallel sentences across 22 Indic languages + English. Downloaded fine.
- ✓ Then, on a hunch, tried the raw Meta URL the tinyurl was *supposed* to
  redirect to: `dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`.
  25.6 MB, `content-type: application/x-tar`. Works, no auth.
- Minor: tarball members are prefixed `./`, so `tf.extractfile("flores200_dataset/...")`
  raised `KeyError`. One-character fix.

**Revision:** what started as a workaround became a better design. I kept
**both** corpora — FLORES (Wikipedia-domain) and IN22 (Indian-domain) — so I
can test whether any finding is a domain artifact rather than asserting it
isn't. That cross-corpus check later became the strongest paragraph in A1.

7 languages × 2036 sentences = 14,252 lines. Alignment guard dropped 0 lines
from either corpus.

---

## Block 3 — A2: measuring the suspects

Built `audit_experiments.py` as a switchable re-implementation of v0's
`analyze()`, with a startup assertion that it reproduces `(1.27, 7.45, 0.226,
1.579, 5.89)` exactly at default settings. Without that assertion no delta is
attributable.

### ✗ Dead end 1: `split(" ")` is real but nearly worthless

```
phantom empty 'words': eng=1 hin=1 (sample)  |  eng=0 hin=6 of 25,649 (FLORES)
sample ratio 5.89 -> 5.92 (+0.6%);  FLORES ratio 6.11 -> 6.11 (+0.0%)
```

I had this drafted as flaw #1. **Measuring it demoted it to a footnote.** The
double spaces in the starter corpora are clearly planted, and I think the plant
is a test of exactly this: whether you report the *magnitude* or just the
existence. On a real corpus it is worth <0.05%. Kept the claim, moved it down,
labelled it "real, immaterial here".

### ✗ Dead end 2: I had `.lower()` backwards

I predicted lowercasing makes English *cheaper* (fewer tokens), inflating the
reported Hindi penalty. Measured:

```
F2 fix: no .lower()   eng 1.287 -> 1.244   ratio 6.11 -> 6.32 (+3.4%)
lines changed by .lower(): eng=1012, hin=38
```

English fertility goes **down** when I remove the lowercasing — so lowercasing
makes English *more expensive*. GPT-2's BPE has dedicated merges for capitalised
forms (`The`, `NASA`, `ISRO`, `Bengaluru`), and folding to lowercase fragments
them. **The bug shrinks the reported gap; v0's headline was conservative, not
inflated.** Fixing v0's code makes Hindi look worse, not better.

This inverted my whole draft framing. I had been writing "the intern
exaggerated"; the truth is the intern's number was too *small* and the
recommendation was wrong for an entirely different reason. Rewrote A2's
narrative around that.

(Surprise worth recording: hin=38 lines changed, because FLORES Hindi carries
Latin-script proper nouns. Even the "no-op on Indic" claim isn't literally true.)

### ✗ Dead end 3: NFC is not the hidden bug

My highest-confidence prediction. Measured across all 14,252 lines, 7 languages,
both corpora:

```
lines where NFC(x) != x:  0 / 14,252     token delta: +0
```

Both corpora already ship NFC. I then built the adversarial case — feed the
pipeline NFD text — and it absorbed it, identical output. Even raw NFC-vs-NFD
token counts differ by **+3 tokens in 200,688 (+0.002%)**, because Devanagari
mostly doesn't decompose; only the nukta forms do, and they're rare.

So NFC is inert *and* correct. It moved from "suspect #4" to a **cleared
control**. This is the one where measurement most clearly saved me from a −5.

### ✓ The real finding: F4/F5, the denominator

Flipping only the "char" denominator, identical tokens:

```
hin-vs-eng, GPT-2, FLORES:
  per byte       2.90x
  per word       6.34x
  per sentence   7.42x
  per grapheme  11.39x
```

**3.9× spread with nothing changed but the divisor.** That is when I understood
what the assignment means by "computes exactly what it says, but what it says is
the wrong thing to compute". The bug isn't in the code; it's that no denominator
here holds *content* constant.

Checked the intuition against the corpus and it held: the same 1012 meanings are
21,901 English words but only **16,100 Kannada words** (0.74×) and 25,649 Hindi
words (1.17×). The word denominator biases in *opposite directions* for Hindi
and Kannada — it flatters exactly the language v0 measured.

Also verified F6 algebraically rather than asserting it:
```python
(T_h/C_h)/(T_e/C_e) == (T_h/W_h)/(T_e/W_e) * (W_h/C_h)/(W_e/C_e)
# 7.4653 == 6.3379 * 1.1779   -> exact, to all printed digits
```
So the "confirming" second metric contributes exactly one new fact — the
chars-per-word ratio — and nothing about the tokenizer.

---

## Block 4 — A3: the tokenizer swap

Hypothesis: report finding 3 ("property of the script, not the tokenizer") is
false. Test: hold corpus and denominator fixed, change only the tokenizer.

```
hin cost vs eng, per parallel sentence, FLORES:
  gpt2      7.42x
  cl100k    4.77x
  muril     1.16x
  sarvam-1  1.13x
```

**Surprise, and bigger than I expected.** I anticipated an Indic tokenizer
landing around 2–3×. It lands at ~1.1× — and per *word*, `muril` gives Hindi
0.99× and `sarvam-1` 0.97×, i.e. **cheaper than English.** ~85% of the measured
"Indic penalty" evaporates on a tokenizer change. Finding 3 isn't imprecise,
it's backwards. (Careful with the causal wording: the swap shows the penalty is
tokenizer-dependent; it does not isolate *which* tokenizer property causes it.)

**Second surprise, which v0 structurally could not have found:** Tamil is
15.54× and Kannada 13.58× on GPT-2 — **~2× worse than Hindi's 7.42×.** v0
measured Hindi, the *best-case* Indic language, and generalised to "all Indic
traffic". A capacity plan built on the Hindi number under-provisions Dravidian
by ~2×. This is the finding that made me add Dravidian languages beyond the
required two, and it is why A1 carries 7 languages instead of 4.

Cross-corpus check (FLORES vs IN22): all conclusions hold within ~3%. Good — the
findings are robust to domain *within edited written prose*, and I said exactly
that and no more in A1's caveats.

Noted honestly in A3: `muril` is an encoder, so it's an existence proof, not a
deployable option; `sarvam-1` is the decoder-shaped comparator. And a 197k vocab
buys token savings at the cost of a bigger embedding/LM-head, which I did not
price.

---

## Block 5 — Part B

**B1.** KV bytes/token was the easy part once I noticed `head_dim × kv_heads =
1024 ≠ d_model = 3072` — GQA, so 8 KV heads, not 24. Using 24 would have
tripled the answer.

✗ **Dead end:** my first pass took "24 GB" literally → 22.08 GiB budget →
**29 sequences**. The log says 25–26 (`32−7 = 25`, `48−23 = 25`, `24/0.93 =
25.8`). A 12% miss, which is too big to wave at.

Rather than fudge the overhead constant to make it fit, I back-solved the memory
from the cleanest row (batch 24, 0.93 util, 0 preemptions):

```
pool = (24 × 4096 × 114,688) / 0.93                  = 11.290 GiB
HBM  = (11.290 + 7.823 + 1.490) / 0.92               = 22.40 GiB
```

22.40 GiB is what an L4 actually exposes (23,028 MiB), not 24 GiB. Redoing B1
with 22.49 GiB gives **26.0 sequences** — right to within one, and the residual
is most likely block-granular allocation (also why util caps at 0.97, never
1.00) — though the spec names neither the engine nor the block size, so I label
that as the leading candidate rather than the established cause. The model of the machine was right; one constant was wrong, and the log
told me which.

**B3.** Guessed `reported_tok_s` included prompt tokens; tested it on all 13
rows rather than one:

```
reported_tok_s == (prompt_len + gen_len) × num_requests / wall_clock_s
max error 0.02% across 13/13 rows
```

✗ **Dead end:** my first "two independent ways" to get goodput were
`gen×n/wall = 200.9` and `reported × 512/4096 = 200.9`. They agree perfectly —
because they are **the same identity**, `reported_tok_s` being defined from
`wall_clock_s`. Not two derivations, one derivation written twice. Nearly shipped
that as evidence.

Replaced the second with one that touches neither column: `batch / itl_p50 =
24 / 0.09607 = 249.8 tok/s`. That **disagreed by 20%**, which was alarming for
about ten minutes until the gap turned out to be exactly prefill (which the ITL
route excludes by construction):

```
decode  = 512 × 96.07 ms = 49.2 s ;  prefill = 61.16 − 49.2 = 12.0 s
86,016 prompt tokens / 12.0 s = 7,185 tok/s
sanity: 2 × 4.2e9 × 86,016 / 12.0 s = 60 TFLOPS = 50% of the L4's 121 peak
```

50% prefill MFU is believable, so the two estimates are *reconciled*, not
averaged. Better answer than the one I set out to write, and it came from the
disagreement.

**B2.** The discriminating column is `itl_ms_p50`, and I nearly missed it. It is
**flat** across the cliff (96.07 → 101.79 → 100.00) while `ttft_ms_p50` explodes
(500 → 637 → 955). Compute saturation would raise ITL with batch. It doesn't —
because the number of sequences actually decoding never exceeds 25. The extra
requests aren't slowing decode down; they aren't running. Wrote B4 around that
specific discriminator, with the falsifying observation stated.

Also checked the report's other claim properly: at equal batch, long prompts are
**worse** on goodput (0.56–0.81×). The report has the sign backwards, not just
the magnitude.

---

## Block 6 — Part C, and a reframe

Started sizing the A100. Data generation ~35 min, LoRA training ~3 h — **well
under a day out of 14 available.**

That killed my initial framing. I had been about to write a memo comparing
paths on GPU cost, which is the obvious axis and the wrong one. **Compute is
~4 h; validation is 30 reviewer-hours covering 2 of 6 languages.** Rewrote the
memo around the reviewer as the binding constraint, which then forced the
uncomfortable-but-correct assumption that four of the six languages cannot
launch in three weeks.

The other thing that fell out: (b) needs the *same* synthetic pairs (a) needs,
plus a second model in the serving path. It is dominated, not merely worse. And
"no external API budget" means our own model must generate the data — which is
what makes the day-1 experiment able to kill all three paths at once.

Power check on the eval, so the threshold isn't decorative: 150 paired
comparisons ⇒ SE = 4.1 pp ⇒ 60% is 2.45σ ⇒ ~80% power at α = 0.05 one-sided.
55% would not be detectable with the reviewer time available, which is why the
threshold is 60%.

---

## Open threads I did not close

Stated here rather than buried, because they are the honest boundary of the
submission:

- **Romanised / code-mixed Indic is unmeasured.** It is the single biggest
  threat to A3's multipliers — romanised Hindi tokenizes roughly like English —
  and I could not size it from a parallel corpus. A4's monitoring metric is
  designed to detect it in production, which is a mitigation, not a measurement.
- **Everything in Part A is input-side.** Serving cost is dominated by output
  tokens. I assume the ratio transfers to generations; untested.
- **The B2 fix is predicted, not run.** No GPU here. I stated the falsifying
  observation (b48 wall clock ~122 s vs ~151 s) so it is checkable in one run.
- **The 197k-vocab trade-off is unpriced** — fewer tokens, bigger embedding and
  LM head. Flagged in A3 §4, not measured.
