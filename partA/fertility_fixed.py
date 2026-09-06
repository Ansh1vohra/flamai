#!/usr/bin/env python3
"""
fertility_fixed.py -- drop-in replacement for starter_kit/fertility.py.

Same CLI shape as v0, so you can run them side by side. Differences, each
justified in AUDIT.md:

  F1  .split()          instead of .split(" ")   -- no phantom empty words
  F2  no .lower()       -- v0's lowercasing is a one-sided edit that only
                          touches Latin script and inflates English fertility
  F3  micro-average     -- sum(tokens)/sum(units), not mean of per-line ratios
  F4  named denominators -- 'chars' is not a thing; you must say which of
                          grapheme cluster / UTF-8 byte / codepoint you mean
  F5  --baseline        -- the reference language is explicit, not argv[0]
  CONCEPT
      --unit sentence   -- default. On a PARALLEL corpus this is the only
                          denominator that holds content constant across
                          languages, which is what a cost question needs.

Kept from v0 because they were correct: NFC normalisation, and
add_special_tokens=False on the HF path.

Usage:
    python fertility_fixed.py \
        --corpus eng=corpus/flores/eng.txt --corpus hin=corpus/flores/hin.txt \
        --tokenizer gpt2 --unit sentence --unit word --baseline eng
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokstats as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, metavar="LANG=PATH")
    ap.add_argument("--tokenizer", default="gpt2",
                    help=f"one of {list(T.TOKENIZERS)}, or hf:<repo_id>")
    ap.add_argument("--unit", action="append", default=None,
                    choices=list(T.DENOMS),
                    help="denominator(s); repeatable. default: sentence + word")
    ap.add_argument("--baseline", default=None,
                    help="reference language for the ratio column "
                         "(default: the first --corpus given, but stated explicitly)")
    ap.add_argument("--macro", action="store_true",
                    help="reproduce v0's mean-of-ratios aggregation (for comparison)")
    a = ap.parse_args()
    units = a.unit or ["sentence", "word"]

    if a.tokenizer.startswith("hf:"):
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.tokenizer[3:])
        encode = lambda s: tk.encode(s, add_special_tokens=False)
    else:
        encode = T.get_encoder(a.tokenizer)

    langs, data = [], {}
    for spec in a.corpus:
        lang, path = spec.split("=", 1)
        langs.append(lang)
        data[lang] = T.read_lines(path)
    base = a.baseline or langs[0]
    if base not in data:
        sys.exit(f"--baseline {base} is not among {langs}")

    n = {len(v) for v in data.values()}
    parallel = len(n) == 1
    if "sentence" in units and not parallel:
        print("WARNING: corpora have different line counts "
              f"({ {k: len(v) for k, v in data.items()} }). 'tok/sentence' is only "
              "meaningful on a line-parallel corpus; treat that column as junk.\n")

    d = {u: T.DENOMS[u] for u in units}
    res = {}
    for lang in langs:
        if a.macro:
            rates = T.macro(data[lang], encode, d)
            tok = sum(len(encode(l)) for l in data[lang])
        else:
            rates, tok, _ = T.micro(data[lang], encode, d)
        res[lang] = (rates, tok)

    agg = "macro (v0-style, mean of per-line ratios)" if a.macro else "micro (sum/sum)"
    print(f"tokenizer: {a.tokenizer}   aggregation: {agg}   baseline: {base}")
    head = f"{'lang':<8}{'lines':>7}{'tokens':>10}"
    for u in units:
        head += f"{'tok/'+u:>15}"
    head += f"{'cost vs '+base:>14}"
    print(head)
    print("-" * len(head))
    for lang in langs:
        rates, tok = res[lang]
        line = f"{lang:<8}{len(data[lang]):>7}{tok:>10}"
        for u in units:
            line += f"{rates[u]:>15.3f}"
        cost = tok / res[base][1] if parallel else float("nan")
        line += f"{cost:>13.2f}x"
        print(line)

    if not parallel:
        print("\n'cost vs baseline' is nan because the corpora are not line-parallel.")
        return
    print(f"\nRatios vs {base}, per denominator "
          f"(identical token counts, different divisors):")
    for u in units:
        print(f"  per {u:<10}" + "  ".join(
            f"{l}={res[l][0][u]/res[base][0][u]:.2f}x" for l in langs if l != base))
    print("\nIf those rows disagree, the denominator is the finding. See AUDIT.md F4.")


if __name__ == "__main__":
    main()
