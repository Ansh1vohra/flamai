# Part B — Capacity reconciliation

**Reproduce every number:** `python partB/capacity.py` (full output saved in
`results.txt`). Nothing below is typed in by hand except the `model_spec.md`
constants at the top of the script.

---

## B1 (a) — KV-cache bytes per token, exactly

```
2 (K and V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes (fp16)
  = 114,688 bytes/token
  = 112 KiB/token
```

The one thing to get right is **8, not 24**. `head_dim × kv_heads = 1024 ≠
d_model = 3072`; this is GQA with 24 query heads sharing 8 KV heads. Using the
query-head count gives 344,064 B/token and over-states KV pressure by exactly 3×.

## B1 (b) — Max concurrent 4096-token sequences

```
KV per full sequence  = 114,688 × 4096 = 469,762,048 B = 0.4375 GiB
weights               = 4.2e9 × 2 B    = 8.4 GB        = 7.823 GiB
non-KV overhead       = 1.6 GB                         = 1.490 GiB
```

The spec says "L4 (24 GB)", which is ambiguous, and the ambiguity is worth ~3
sequences — so I computed both rather than picking one:

| assumed total HBM | budget (×0.92) | KV pool | **sequences** |
|---|---|---|---|
| 24 GiB (naive) | 22.080 GiB | 22.080 − 7.823 − 1.490 = 12.767 GiB | 12.767 / 0.4375 = **29.2 → 29** |
| 22.49 GiB (what an L4 actually reports, 23028 MiB) | 20.689 GiB | 11.376 GiB | 11.376 / 0.4375 = **26.0 → 26** |

**Prediction: 26 sequences, with 29 as the upper bound if "24 GB" were literal.**

## B1 (c) — Check against the log

The `prompt_len=3584, gen_len=512` rows hold exactly 4096 tokens per sequence,
so `kv_cache_util` maps directly onto sequences resident.

| batch | preempted | resident | kv_util | implied capacity (batch/util) |
|---|---|---|---|---|
| 4 | 0 | 4 | 0.16 | 25.0 |
| 8 | 0 | 8 | 0.31 | 25.8 |
| 16 | 0 | 16 | 0.62 | 25.8 |
| 24 | 0 | 24 | 0.93 | 25.8 |
| 32 | **7** | **25** | 0.97 | — |
| 48 | **23** | **25** | 0.97 | — |

Three independent readings agree: `32 − 7 = 25`, `48 − 23 = 25`, and
`24 / 0.93 = 25.8`. **The machine holds ~25–26 full-length sequences.**

My 26-sequence prediction is right to within one sequence. The 29 is 12% high.

**Where the gap goes.** Back-solving the memory from the cleanest row (batch 24:
0.93 util, 0 preemptions):

```
KV in use   = 24 × 4096 × 114,688 B                        = 10.500 GiB
pool size   = 10.500 / 0.93                                = 11.290 GiB
implied HBM = (11.290 + 7.823 + 1.490) / 0.92              = 22.40 GiB
```

**22.40 GiB** — which is the real usable capacity of an L4, not 24 GiB. The
entire discrepancy is the marketing "24 GB" versus the ~22.5 GiB the driver
actually exposes. So the model of the machine is correct; only the memory
constant was wrong, and the log tells us what it should have been.

*(Residual ~1 sequence: PagedAttention allocates in blocks of 16 tokens, so a
partly-filled block is charged in full, and `kv_cache_util` is reported at block
granularity — which is also why util tops out at 0.97, never 1.00.)*

---

## B2 — The long-context throughput anomaly

| batch | wall_s | reported_tok_s | **goodput** | ttft_p50 | **itl_p50** | e2e_p95 | **preempted** | kv_util |
|---|---|---|---|---|---|---|---|---|
| 4 | 28.98 | 565.4 | 70.7 | 483.2 | 51.33 | 32673 | 0 | 0.16 |
| 8 | 36.30 | 902.6 | 112.8 | 519.0 | 62.26 | 39983 | 0 | 0.31 |
| 16 | 49.97 | 1311.4 | 163.9 | 498.3 | 77.20 | 54602 | 0 | 0.62 |
| **24** | **61.16** | **1607.4** | **200.9** | 500.5 | 96.07 | 69221 | 0 | 0.93 |
| 32 | 94.71 | 1384.0 | 173.0 | 636.9 | 101.79 | 97466 | **7** | 0.97 |
| 48 | 151.41 | 1298.5 | 162.3 | 955.4 | 100.00 | 105428 | **23** | 0.97 |

### The anomaly

Throughput is **not monotone in batch size**. It peaks at batch 24 and then
*falls*: 1607.4 → 1384.0 → 1298.5 (and goodput 200.9 → 173.0 → 162.3). Adding
24 more concurrent requests made the server **20% slower in absolute terms**.
Naive "throughput scales with batch" predicts monotone increase; the log shows a
cliff between batch 24 and 32.

### The mechanism, column by column

- **`kv_cache_util` saturates**: 0.62 → 0.93 → **0.97 → 0.97**. It stops rising
  because there is no more pool. Batches 32 and 48 are pinned at the ceiling.
- **`preempted_seqs` steps 0 → 0 → 7 → 23** at exactly that point, and the
  arithmetic closes with B1: `32 − 7 = 25`, `48 − 23 = 25`. The scheduler admits
  the 25 sequences that fit and evicts the rest. This is KV-cache exhaustion,
  not compute saturation.
- **`itl_ms_p50` is FLAT across the cliff**: 96.07 → 101.79 → 100.00. *This is
  the decisive column.* If the GPU were compute-bound, per-token latency would
  rise with the number of sequences in the batch. It does not — because the
  number of sequences actually decoding never rises above 25. The extra requests
  are not slowing decode down; **they are not running**.
- **`ttft_ms_p50` explodes instead**: 500.5 → 636.9 → **955.4**. Preempted
  sequences are recomputed from scratch on resume (vLLM's RECOMPUTE mode
  re-prefills the whole 3584-token prompt). That wasted work lands on time-to-
  first-token and on queueing, not on inter-token latency — exactly the
  signature the ITL/TTFT split shows.
- **`wall_clock_s` prices the waste**: batch 48 takes 151.41 s for 2× the
  requests of batch 24 (61.16 s). Perfect two-wave serialisation would be
  122.32 s. The extra **29.09 s (+24%)** is the recompute-and-thrash cost of
  admitting work the GPU has no memory for.

### Proposed change, with a predicted number

**Cap admission at measured capacity: `--max-num-seqs 24`** (vLLM default is
256). Queue the excess instead of admitting and evicting it.

Predicted, for the batch-48 workload:

| | now | predicted | Δ |
|---|---|---|---|
| wall clock | 151.41 s | ~122.3 s | **−19%** |
| goodput | 162.3 tok/s | ~200.9 tok/s | **+24%** |
| preemptions | 23 | 0 | — |
| p95 e2e | 105.4 s | **worse** for the queued half (+~61 s wait) | trade-off |

That last row is the honest cost: this buys throughput with tail latency. Worth
it for batch offline work, possibly not for interactive traffic.

**Falsifiable:** if batch 48 with `max_num_seqs=24` lands near 151 s rather than
~122 s, preemption is not the mechanism and I am wrong.

**Better fix if we can spend the engineering time:** the root constraint is
112 KiB/token × 4096 = 0.4375 GiB per sequence on a 22.4 GiB card. **fp8 KV
cache** halves bytes/token to 57,344, roughly doubling resident sequences to
~50 and moving the cliff past batch 48 entirely — predicted goodput ~400 tok/s
at batch 48 with 0 preemptions. The accuracy cost of fp8 KV must be measured
separately and is not free.

---

## B3 — The misread column

### What it is: `reported_tok_s`

`reported_tok_s` counts **prompt + generated** tokens. It is a total-token
counter, not a generation-throughput counter. Tested on all 13 rows:

```
reported_tok_s  ==  (prompt_len + gen_len) × num_requests / wall_clock_s
```

Max error across the whole log: **0.02%**. Confirmed.

Both of REPORT_v0's Section 2 conclusions fall out of that one misreading:

**"Longer prompts give better throughput."** Prompt tokens are prefill —
processed once, in parallel, in a single forward pass. Padding the prompt from
512 to 3584 tokens multiplies the *numerator* of `reported_tok_s` by 5.3×
without generating a single extra token for a user. The metric goes up; nothing
good happened. Compare like-for-like on generated tokens:

| batch | goodput, 512-prompt | goodput, 3584-prompt | ratio |
|---|---|---|---|
| 4 | 87.0 | 70.7 | 0.81 |
| 8 | 165.2 | 112.8 | 0.68 |
| 16 | 294.5 | 163.9 | **0.56** |

Long prompts are **worse** at every batch size — up to 1.8× worse. **The report
has the sign backwards.**

**"Batch 48 will deliver ~3200 tok/s."** Linear extrapolation of a
prompt-inflated number, projected straight through the KV cliff. Measured
`reported_tok_s` at batch 48 is **1298.5 — 59% below the claim**, and *lower*
than batch 24. On the honest basis, goodput is 162.3 tok/s, also below batch 24.

### Honest goodput of the batch-24 long-prompt row

**(A) From counts and wall clock**

```
512 gen × 24 requests / 61.16 s = 200.9 tok/s
```
Equivalently, from the report's own column: `1607.4 × 512/4096 = 200.9 tok/s`.

**(B) From inter-token latency — uses neither `wall_clock_s` nor the counter**

```
batch / itl_p50 = 24 / 0.09607 s = 249.8 tok/s   (steady-state decode only)
```

*(A) and (B) are genuinely independent — (B) touches no column that (A) uses.*
They differ by 20%, and **the difference is exactly prefill**, which (B)
excludes by construction:

```
decode  = 512 steps × 96.07 ms                = 49.2 s
prefill = 61.16 − 49.2                        = 12.0 s
        → 86,016 prompt tokens / 12.0 s       = 7,185 tok/s prefill
sanity  : 2 × 4.2e9 × 86,016 = 7.2e14 FLOP
          7.2e14 / 12.0 s = 60 TFLOPS = 50% of the L4's 121 TFLOPS peak
```

50% prefill MFU is a believable number for a dense 4B on an L4, so the
decomposition holds and the two estimates are reconciled rather than merely
averaged.

**Answer: 200.9 tok/s end-to-end** (249.8 tok/s if you count only the decode
phase). Against the 1607.4 in the deck, that is an **8× overstatement**.

### What the report should have said

> At batch 24 with 3584-token prompts the server sustains **~200 generated
> tok/s** (~8.4 tok/s per sequence). The `reported_tok_s` column includes prompt
> tokens and must not be used for capacity planning. Generated throughput is
> **better with short prompts** — ~295 tok/s at batch 16 with 512-token prompts,
> 1.8× the long-prompt figure — so clients should be encouraged to send *less*
> context, not more. Throughput does **not** scale linearly with batch: it peaks
> at batch 24 and regresses beyond it as the KV cache saturates at ~25 resident
> sequences. Plan capacity at **~200 tok/s per L4** for long-context traffic and
> **~300 tok/s** for short-context, and treat batch 24 as the ceiling.

---

## B4 — The counter to pull

Pull **`vllm:num_preemptions_total`** — the monotonic engine counter behind the
`Sequence group … is preempted by PreemptionMode.RECOMPUTE` log line.

Expected, if my B2 mechanism is right: **flat at 0** for the entire batch ≤ 24
sweep including the full 61.16 s batch-24 run, then stepping to **~7** during the
batch-32 run and **~23 more** during batch-48 — reproducing the log's
`preempted_seqs` column exactly. Alongside it, `vllm:gpu_cache_usage_perc`
should sit pinned at **~0.97** (never 1.00, because blocks are allocated in
16-token granules) and `vllm:num_requests_waiting` should be **> 0** throughout.

The discriminating observation is **`vllm:num_requests_running`, which should
plateau at ~25 rather than reaching 32 or 48.** That is what separates my
explanation from the alternative one. If the slowdown were compute saturation
rather than KV eviction, `num_requests_running` would read 32 and 48 with
preemptions still at 0, and `itl_ms_p50` would rise proportionally with batch
instead of sitting flat at ~100 ms. So: **running ≈ 25 with preemptions > 0
confirms me; running = 48 with preemptions = 0 falsifies me.**
