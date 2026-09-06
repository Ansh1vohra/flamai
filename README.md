# The Audit — submission

Audit of `REPORT_v0.md`, its tokenizer script, and its serving conclusions.

## The three findings, in one paragraph each

**Part A.** `fertility.py` has two real code bugs (`split(" ")`, `.lower()`) and
one methodology bug (mean-of-ratios), but together they move the headline by only
**+3.7%** — and in the *opposite* direction to the report's error: correcting
them makes Hindi look **worse**, not better. The actual problem is conceptual.
Neither `tok/word` nor `tok/char` holds content constant across languages, so the
"Indic penalty" reads **2.90×, 6.34×, 7.42× or 11.39×** from one identical set of
token counts, depending only on the divisor. And the report's causal claim —
"a property of the script, not the tokenizer" — is false: holding corpus and
metric fixed and swapping only the tokenizer moves Hindi from **7.42× to 1.13×**.
The recommendation should be *change the tokenizer*, not *budget 6×*. Separately,
Tamil (15.54×) and Kannada (13.58×) are ~2× worse than Hindi, so "all Indic
traffic" is not one bucket — a finding v0's Hindi-only corpus could not produce.

**Part B.** KV cache is **114,688 B/token** (112 KiB); the GPU holds **~26**
concurrent 4096-token sequences, which the log confirms at 25–26 (`32−7 = 25`,
`48−23 = 25`). The long-context throughput cliff at batch 32 is KV-cache
exhaustion — `itl_ms_p50` stays **flat** while `ttft_ms_p50` explodes and
`preempted_seqs` steps 0 → 7 → 23. And REPORT_v0 §2 misreads one column:
`reported_tok_s` counts **prompt + generated** tokens (verified on 13/13 rows to
0.02%), so "long prompts are faster" has the **sign backwards** and the batch-48
projection is 59% high. Honest goodput of the batch-24 row is **200.9 tok/s**,
not 1607.4.

**Part C.** The binding constraint is not the A100 (~4 GPU-hours of 336) but the
reviewer (30 hours, covering 2 of 6 languages). Recommend LoRA SFT scoped to
Hindi + Kannada, with prompt-engineering as a mandatory week-1 baseline that can
kill the SFT before it starts.

## Layout

```
NOTEBOOK.md          chronological log — hypotheses, dead ends, reversals
AI_USAGE.md          where AI helped and where it misled me
partA/
  CORPUS.md          A1 — corpus, preprocessing, what it cannot tell us
  AUDIT.md           A2 — every claimed flaw, isolated and measured
  ANALYSIS.md        A3 — corrected analysis + denominator reasoning
  MEMO.md            A4 — recommendation memo
  build_corpus.py    downloads FLORES-200 + IN22-Gen, writes corpus/
  tokstats.py        shared measurement core (tokenizers, denominators)
  audit_experiments.py  A2 ablations   -> results/a2_audit.{txt,json}
  controls.py        the three cleared suspects
  analyze.py         A3 grid          -> results/a3_analysis.{md,csv}
  fertility_fixed.py corrected drop-in replacement for fertility.py
partB/
  ANSWERS.md         B1–B4 written answers
  capacity.py        all Part B arithmetic
  results.txt        its saved output
partC/memo.md        C — decision memo
starter_kit/         the original, unmodified
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python partA/build_corpus.py        # ~1 min, downloads ~26 MB
python partA/audit_experiments.py   # A2  (asserts it reproduces REPORT_v0's table)
python partA/controls.py            # A2  cleared suspects
python partA/analyze.py             # A3  ~2 min, downloads 2 HF tokenizers
python partB/capacity.py            # B   pure arithmetic, instant
```

`audit_experiments.py` **asserts on startup** that its re-implementation of v0
reproduces the published `(1.27, 7.45, 0.226, 1.579, 5.89)` exactly. If that
assertion ever fails, every delta in A2 is void — which is the point of it.

Ad-hoc runs against any corpus or tokenizer:

```bash
python partA/fertility_fixed.py \
  --corpus eng=partA/corpus/flores/eng.txt \
  --corpus tam=partA/corpus/flores/tam.txt \
  --tokenizer hf:google/muril-base-cased \
  --unit sentence --unit word --unit byte --baseline eng
```

## Claims I am deliberately *not* making

Listed here so they are not mistaken for oversights — details in the docs:

- `random.seed(1337)`, `unicodedata.normalize("NFC")` and
  `add_special_tokens=False` **are not bugs.** Each was on my suspect list and
  each was cleared by measurement ([AUDIT.md](partA/AUDIT.md) C5–C7).
- `split(" ")` is a real bug worth **<0.05%** on a real corpus. Claimed, but not
  led with.
- Everything in Part A is **input-side** and on **edited written prose**.
  Romanised and code-mixed Indic — the biggest threat to these multipliers — is
  unmeasured; see [CORPUS.md](partA/CORPUS.md) and the monitoring metric in
  [MEMO.md](partA/MEMO.md).
- The Part B config fix is **predicted, not run** (no GPU here). The falsifying
  observation is stated in [ANSWERS.md](partB/ANSWERS.md) B2.
