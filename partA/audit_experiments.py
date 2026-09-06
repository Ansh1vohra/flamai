#!/usr/bin/env python3
"""
audit_experiments.py -- A2. One experiment per claimed flaw in fertility.py.

Method: `v0_metric` is a faithful re-implementation of fertility.py's
analyze(), with each individual behaviour behind a switch.  Test 0 asserts
that at default switch settings it reproduces the shipped script's numbers
to the printed precision -- so every delta below is attributable to the one
switch that was flipped, and nothing else.

Run:
    python audit_experiments.py                 # sample corpus + FLORES + IN22
    python audit_experiments.py --corpus flores
"""
import argparse, json, os, subprocess, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokstats as T

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KIT = os.path.abspath(os.path.join(HERE, "..", "starter_kit"))


# ------------------------------------------------------ v0, with switches
def v0_metric(lines, encode, *, lowercase=True, word_split=" ",
              aggregation="macro", char_unit="codepoint"):
    """fertility.py::analyze() with each behaviour switchable.

    Defaults == the shipped script:
      lowercase=True        line = line.lower()
      word_split=" "        words = line.split(" ")
      aggregation="macro"   mean of per-line ratios
      char_unit="codepoint" chars = len(line)
    """
    charfn = T.DENOMS[char_unit]
    fert, tpc = [], []
    tt = tw = tc = 0
    for line in lines:
        s = line.lower() if lowercase else line
        t = len(encode(s))
        w = len(s.split(word_split) if word_split else s.split())
        c = charfn(s)
        fert.append(t / w); tpc.append(t / c)
        tt += t; tw += w; tc += c
    if aggregation == "macro":
        return sum(fert) / len(fert), sum(tpc) / len(tpc)
    return tt / tw, tt / tc


def pair(eng, hin, encode, **kw):
    ef, et = v0_metric(eng, encode, **kw)
    hf, ht = v0_metric(hin, encode, **kw)
    return dict(eng_fert=ef, hin_fert=hf, fert_ratio=hf / ef,
                eng_tpc=et, hin_tpc=ht, tpc_ratio=ht / et)


def row(name, r, base=None):
    d = ""
    if base:
        d = (f"  | fert_ratio {base['fert_ratio']:.2f} -> {r['fert_ratio']:.2f} "
             f"({(r['fert_ratio']/base['fert_ratio']-1)*100:+.1f}%)")
    return (f"{name:<44}{r['eng_fert']:7.3f}{r['hin_fert']:8.3f}"
            f"{r['fert_ratio']:8.2f}{r['eng_tpc']:9.3f}{r['hin_tpc']:8.3f}"
            f"{r['tpc_ratio']:8.2f}{d}")


HDR = (f"{'experiment':<44}{'eng_f':>7}{'hin_f':>8}{'ratio':>8}"
       f"{'eng_tpc':>9}{'hin_tpc':>8}{'ratio':>8}")


def run_corpus(label, eng, hin, encode, out):
    p = lambda s: (print(s), out.append(s))
    p(f"\n{'='*118}\nCORPUS: {label}  (eng {len(eng)} lines / hin {len(hin)} lines)"
      f"\n{'='*118}")
    p(HDR); p("-" * 118)
    base = pair(eng, hin, encode)
    p(row("V0  as shipped", base))

    # -- F1: split(" ") instead of .split()
    r = pair(eng, hin, encode, word_split=None)
    p(row("F1  fix: .split() on whitespace", r, base))
    ph_e = sum(s.lower().split(" ").count("") for s in eng)
    ph_h = sum(s.lower().split(" ").count("") for s in hin)
    p(f"    -> phantom empty 'words' created by split(' '): eng={ph_e} hin={ph_h} "
      f"(of {sum(len(s.split(' ')) for s in eng)}/{sum(len(s.split(' ')) for s in hin)} counted)")

    # -- F2: .lower()
    r = pair(eng, hin, encode, lowercase=False)
    p(row("F2  fix: no .lower()", r, base))
    lo_e = sum(1 for s in eng if s.lower() != s)
    lo_h = sum(1 for s in hin if s.lower() != s)
    p(f"    -> lines actually changed by .lower(): eng={lo_e} hin={lo_h} "
      f"(the operation is a no-op on Indic scripts -- it is a one-sided edit)")

    # -- F3: macro vs micro aggregation
    r = pair(eng, hin, encode, aggregation="micro")
    p(row("F3  fix: micro-average (sum/sum)", r, base))

    # -- F4: char denominator is not a character
    for u in ("grapheme", "byte"):
        r = pair(eng, hin, encode, char_unit=u)
        p(row(f"F4  chars -> {u}s", r, base))

    # -- all code fixes together
    allfix = pair(eng, hin, encode, lowercase=False, word_split=None,
                  aggregation="micro", char_unit="grapheme")
    p(row("F1+F2+F3+F4 all code fixes", allfix, base))

    # -- F6 (control): NFC normalisation -- claimed HARMLESS
    def strip_nfc(ls):
        return [unicodedata.normalize("NFD", l) for l in ls]  # worst case: feed NFD
    same_e = sum(1 for s in eng if unicodedata.normalize("NFC", s) != s)
    same_h = sum(1 for s in hin if unicodedata.normalize("NFC", s) != s)
    r_nfd = pair(strip_nfc(eng), strip_nfc(hin), encode)
    p(row("F6  CONTROL: same text fed as NFD", r_nfd, base))
    p(f"    -> lines where NFC(x) != x in our corpus: eng={same_e} hin={same_h}. "
      f"NFC is a measured no-op here AND it is the safe direction (see AUDIT.md F6).")

    return dict(base=base, all_code_fixes=allfix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--corpus", default="all",
                    choices=["all", "sample", "flores", "in22"])
    a = ap.parse_args()
    encode = T.get_encoder(a.tokenizer)
    out, results = [], {}

    print(f"tokenizer: {a.tokenizer} (same as REPORT_v0 Section 1)")
    out.append(f"tokenizer: {a.tokenizer} (same as REPORT_v0 Section 1)")

    # Test 0 -- prove the harness reproduces the shipped script exactly.
    se = T.read_lines(os.path.join(KIT, "corpus_sample", "eng_sample.txt"))
    sh = T.read_lines(os.path.join(KIT, "corpus_sample", "hin_sample.txt"))
    b = pair(se, sh, encode)
    got = (round(b["eng_fert"], 2), round(b["hin_fert"], 2),
           round(b["eng_tpc"], 3), round(b["hin_tpc"], 3), round(b["fert_ratio"], 2))
    want = (1.27, 7.45, 0.226, 1.579, 5.89)
    assert got == want, f"harness does NOT reproduce v0: {got} != {want}"
    msg = (f"TEST 0 PASS: harness reproduces REPORT_v0 Table 1 exactly "
           f"({got} == published {want})")
    print(msg); out.append(msg)

    sets = []
    if a.corpus in ("all", "sample"):
        sets.append(("sample (starter_kit, 10 sentences)", se, sh))
    for c in ("flores", "in22"):
        if a.corpus in ("all", c):
            sets.append((f"{c} (A1 eval corpus)",
                         T.read_lines(T.corpus_path(c, "eng")),
                         T.read_lines(T.corpus_path(c, "hin"))))
    for label, e, h in sets:
        results[label.split()[0]] = run_corpus(label, e, h, encode, out)

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "a2_audit.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RES, "a2_audit.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RES}/a2_audit.txt and a2_audit.json")


if __name__ == "__main__":
    main()
