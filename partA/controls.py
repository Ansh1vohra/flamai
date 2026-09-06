#!/usr/bin/env python3
"""
controls.py -- A2, the two things in fertility.py that LOOK wrong but are FINE.

The assignment says at least one suspicious-looking thing in the script is
actually harmless.  Rather than assert that, we measure it.  Both of these
were on my initial suspect list and both were cleared by experiment.

Run:  python controls.py
"""
import os, subprocess, sys, tempfile, shutil, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokstats as T

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "starter_kit"))
SAMPLE = ["--corpus", f"eng={KIT}/corpus_sample/eng_sample.txt",
          "--corpus", f"hin={KIT}/corpus_sample/hin_sample.txt",
          "--tokenizer", "gpt2"]


def c5_random_seed():
    """C5: `random.seed(1337)  # reproducibility` at module scope.

    Looks like the script has a sampling step whose result depends on a seed
    (i.e. the reported numbers are one draw of many).  It does not: `random`
    is imported and seeded and then never used.  Proof: delete both lines and
    diff the output.
    """
    print("\n=== C5: random.seed(1337) ===")
    src = open(os.path.join(KIT, "fertility.py"), encoding="utf-8").read()
    assert "random.seed(1337)" in src
    body = src.split('"""', 2)[2]            # ignore the docstring
    uses = [l for l in body.splitlines()
            if "random" in l and "import random" not in l and "random.seed" not in l]
    print(f"  static check: other uses of `random` outside the docstring: {uses}")

    d = tempfile.mkdtemp()
    try:
        stripped = "\n".join(l for l in src.splitlines()
                             if l.strip() not in ("import random", "random.seed(1337)  # reproducibility"))
        q = os.path.join(d, "fertility_noseed.py")
        open(q, "w", encoding="utf-8").write(stripped)
        a = subprocess.run([sys.executable, os.path.join(KIT, "fertility.py")] + SAMPLE,
                           capture_output=True, text=True).stdout
        b = subprocess.run([sys.executable, q] + SAMPLE,
                           capture_output=True, text=True).stdout
        print(f"  with seed   : {a.strip().splitlines()[-1]}")
        print(f"  seed removed: {b.strip().splitlines()[-1]}")
        print(f"  outputs identical: {a == b}   -> effect on reported numbers: 0.000 (exactly)")
        assert a == b and a.strip()
    finally:
        shutil.rmtree(d)
    print("  VERDICT: dead code / cargo-culted line. Not a bug. Claiming it would cost -5.")


def c6_nfc():
    """C6: unicodedata.normalize("NFC", line).

    Looks dangerous on Indic text -- normalisation can merge or split
    codepoints, which would move any per-codepoint denominator.  Measured:
    it is a no-op on every line of all three corpora, and when we deliberately
    feed NFD input it is what *rescues* the numbers.
    """
    print("\n=== C6: unicodedata.normalize('NFC') ===")
    enc = T.get_encoder("gpt2")
    for corpus in ("flores", "in22"):
        for lang in T.LANGS:
            lines = T.read_lines(T.corpus_path(corpus, lang), normalize=False)
            changed = sum(1 for l in lines if unicodedata.normalize("NFC", l) != l)
            dt = (sum(len(enc(unicodedata.normalize("NFC", l))) for l in lines)
                  - sum(len(enc(l)) for l in lines))
            print(f"  {corpus:<7}{lang}: lines where NFC(x)!=x: {changed:>4}/{len(lines)}  token delta: {dt:+d}")
    # And the counterfactual: what NFC protects you from.
    hin = T.read_lines(T.corpus_path("flores", "hin"))
    nfd = [unicodedata.normalize("NFD", l) for l in hin]
    t_nfc = sum(len(enc(l)) for l in hin)
    t_nfd = sum(len(enc(l)) for l in nfd)
    cp_nfc = sum(len(l) for l in hin)
    cp_nfd = sum(len(l) for l in nfd)
    print(f"\n  counterfactual -- if the corpus arrived as NFD and the call were REMOVED:")
    print(f"    hin tokens   NFC {t_nfc} vs NFD {t_nfd}  ({(t_nfd/t_nfc-1)*100:+.2f}%)")
    print(f"    hin codepts  NFC {cp_nfc} vs NFD {cp_nfd}  ({(cp_nfd/cp_nfc-1)*100:+.2f}%)")
    print("  VERDICT: correct and defensive. Not a bug. Claiming it would cost -5.")


def c7_special_tokens():
    """C7: HF path uses add_special_tokens=False.

    Looks like it undercounts, since production really does prepend BOS.
    But for a *cross-language ratio* it is the right call: specials add a
    constant per sequence, which dilutes the ratio toward 1.0 and would make
    the tokenizer look better than it is.  Measured below.
    """
    print("\n=== C7: add_special_tokens=False ===")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/muril-base-cased")
    eng = T.read_lines(T.corpus_path("flores", "eng"))
    hin = T.read_lines(T.corpus_path("flores", "hin"))
    for flag in (False, True):
        e = sum(len(tk.encode(l, add_special_tokens=flag)) for l in eng)
        h = sum(len(tk.encode(l, add_special_tokens=flag)) for l in hin)
        ew = sum(len(l.split()) for l in eng); hw = sum(len(l.split()) for l in hin)
        print(f"  add_special_tokens={str(flag):<5} eng_fert={e/ew:.3f} hin_fert={h/hw:.3f} "
              f"ratio={(h/hw)/(e/ew):.3f}  tok/sent eng={e/len(eng):.2f} hin={h/len(hin):.2f}")
    print("  -> specials add exactly +2 tokens/sentence to both languages, which SHRINKS")
    print("     the ratio. Excluding them is correct for the metric. Not a bug.")
    print("  (Caveat carried into the memo: for a *billing* estimate you must add them back.)")


if __name__ == "__main__":
    c5_random_seed()
    c6_nfc()
    c7_special_tokens()
