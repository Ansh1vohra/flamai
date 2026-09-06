# Part C — Decision memo: casual-register Indic replies

**Recommendation: (a) LoRA SFT on synthetic casualized pairs — but launch-scoped
to Hindi and Kannada only — with (c) prompt-engineering run first as a
mandatory week-1 baseline and standing fallback. Reject (b).**

The binding constraint is **not the GPU. It is the reviewer.** Compute is
~4 h of the 336 h we have; validation is 30 h covering 2 of 6 languages. Every
decision below follows from that.

**Why reject (b):** a ≤1B rewriter must be trained on the same synthetic
casualized pairs that (a) needs, so under these assumptions it costs everything
(a) costs *plus* a second model in the serving path — roughly 2× latency, since
it regenerates the full reply, and Indic replies are token-heavy. That makes it
clearly less attractive than (a) here, not merely worse on one axis. It becomes
attractive again if adapter deployment is blocked. It also puts the hardest part of the
job — fluent register control in six Indic languages — on the weakest model
available. Reconsider only if adapter deployment is blocked.

## Assumptions (labelled, because several are load-bearing)

1. Base model is a ~7–8B open Indic-capable instruct model we already serve, and
   we can ship a LoRA adapter without a full redeploy.
2. "No external API budget" ⇒ **synthetic data must come from our own model on
   our own A100.** This is the assumption that makes the day-1 experiment
   decisive: if our model cannot produce good casual Indic text under generous
   prompting, it cannot generate the training data either, and (a) *and* (b) both
   die at once.
3. Casualness is judgeable by blind pairwise preference; absolute Likert is not
   reliable across one rater.
4. Eval prompts are sampled from real production traffic, not written fresh.
5. **Tamil, Telugu, Bengali, Marathi do not launch in 3 weeks.** We have no
   reviewer for them, and shipping a register change into a language nobody on
   the team can read is how you find out from users. Train them, hold them
   behind a flag, ship when a reviewer exists.

## Back-of-envelope arithmetic

*Every figure in this section is a **planning assumption**, not a measurement —
I have not benchmarked this stack. They are order-of-magnitude inputs to a
go/no-go decision, and the conclusion I draw from them (compute is not the
constraint) survives all of them being wrong by 3×, which is why I am
comfortable acting on them.*

**Data volume.** 4k pairs × 6 languages = 24k pairs, ~400 tokens each ≈ 10M
tokens. (I assume 2–4k pairs/language is roughly where LoRA style transfer
saturates — a prior from published style/register-transfer work, not something I
measured; more data buys diversity, not register. Worth a day-1 sanity check.)

**Generation cost.** 24k × ~200 output tokens = 4.8M output tokens. Assuming
vLLM on one A100-80GB with an 8B model sustains ~2.5k output tok/s → **~35 min**,
call it 3 h with few-shot prefill, rejection sampling and retries.

**Training cost.** LoRA, bf16, 3 epochs over 10M tokens = 30M tokens; assuming
~3k tok/s on one A100 → **~3 h**. Two or three sweeps still fits inside a day.

**⇒ Total GPU: well under a day of the 14 available.** The A100 is not scarce.
Spend the slack on rejection sampling and on 3–4 adapter variants.

**Reviewer throughput.** 10 h/week × 3 weeks = **30 h, Hindi + Kannada only**.
Blind A/B on one prompt ≈ 1.5 min ⇒ ~40 items/h. Budget:

| week | reviewer spend |
|---|---|
| 1 | 2 h day-1 triage + 3 h authoring ~40 gold casual exemplars/language + 5 h baseline eval |
| 2 | 10 h QA on a 300-pair sample of the synthetic data (the highest-leverage 10 h we have) |
| 3 | 7.5 h final blind eval (150 prompts × 2 languages) + 2.5 h slack |

150 paired comparisons per language gives SE = 4.1 pp on a win rate, so a true
60% is 2.45σ from chance — **~80% power at α = 0.05 one-sided.** That is a normal
approximation to the binomial under the stated design assumptions (independent
items, no rater drift), i.e. a property of the *plan*, not a measured property of
an evaluation that has been run. It is why the threshold below is 60% and not
55%: 55% is not detectable with the reviewer time we have.

## Success metric (numeric)

**≥ 60% blind pairwise win rate vs current production**, in *both* Hindi and
Kannada independently, on 150 held-out real-traffic prompts per language, with
the reviewer answering one question: *"which reply sounds like a person talking
rather than a textbook?"* (ties split).

**Guardrail, and it is not optional:** ≤ 2% of candidate replies newly flagged as
factually wrong, instruction-violating, or inappropriately over-familiar
(wrong T-V/honorific level for the context — the specific failure mode of
casualization in Hindi and Kannada). Register wins that cost correctness are not
wins, and casualization is exactly the change that trades one for the other.

## Kill criterion

Three, each with a date:

- **End of week 1** — if prompt-engineering alone (path c) already clears 60%,
  **kill the SFT and ship prompting.** Do not spend two weeks buying an
  improvement we already have.
- **End of week 1** — if our own model cannot produce casual Indic text the
  reviewer rates as acceptable even with best-of-10 sampling and 5-shot gold
  exemplars, **kill (a) and (b) both**, because we cannot build the training set
  without an external API. Fall back to prompting + a hand-written exemplar bank.
- **End of week 2** — if the first SFT adapter is below **55%** win rate on
  Hindi, or breaches the 2% guardrail, **kill path (a)** and ship prompting for
  the launch review. Week 3 is reserved for validating a decision, not for
  rescuing one.

## Day-1 experiment

**Cost: ~3 GPU-hours and 2 reviewer-hours. It discriminates between all three
paths before we commit to any.**

Sample 50 real Hindi and 50 real Kannada prompts from production logs. Generate
four conditions with the current model: **(0)** current production prompt;
**(1)** + a casual-register system prompt; **(2)** + 5 native-authored casual
few-shot exemplars; **(3)** condition 2 with best-of-10 sampling, reviewer-blind.
Reviewer rates 0-vs-1, 0-vs-2, 0-vs-3 blind and shuffled (~2 h).

What each outcome buys:

- **Condition 1 or 2 clears 60%** → prompting is sufficient. Ship (c) in week 1,
  kill the SFT, spend the remaining reviewer hours extending to more languages
  and hardening against regression. *Cheapest possible win.*
- **Condition 3 clears 60% but 1 and 2 do not** → **the expected outcome, and
  the case for (a).** It says the model *can* produce the target register when
  we spend 10× compute and a long prompt — affordable offline as a data
  generator, not affordable per request. That is exactly the asymmetry SFT
  exists to exploit: distil the expensive condition-3 behaviour into weights, so
  serving pays nothing extra per request (which also matters given the Part A
  finding that Indic prompt tokens are the expensive ones).
- **Nothing clears 60%** → the model has no casual Indic register to elicit.
  All three paths are blocked on data we cannot generate; escalate for either a
  reviewer-authored seed set or an API exception, and do not promise a 3-week
  launch.
