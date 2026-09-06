#!/usr/bin/env python3
"""
build_corpus.py -- A1: assemble the multilingual eval corpus.

Two parallel corpora, both human-translated, sentence-aligned line-by-line:

  flores  FLORES-200 devtest (NLLB / Meta).  Wikipedia-derived prose,
          translated into 200+ languages.  1012 lines.
  in22    IN22-Gen (AI4Bharat), via the ungated mteb/IN22-Gen mirror.
          1024 lines drawn from Indian-context sources across 13 domains.

We keep BOTH because they differ in domain.  If a tokenizer conclusion
flips between them, the conclusion is about the domain, not the tokenizer
-- and we want to be able to see that.

Languages: eng, hin + three Dravidian (kan, tam, tel) + ben, mar.
(The assignment requires >=4 incl. English, Hindi and two Dravidian; we
carry seven because Part C names exactly these six Indic languages.)

Usage:
    python build_corpus.py                  # downloads + writes corpus/
    python build_corpus.py --check          # verify existing corpus only
"""
import argparse, hashlib, io, json, os, sys, tarfile, unicodedata, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")
CACHE = os.path.join(OUT, "_cache")

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# short code -> FLORES-200 code (which is also the IN22 column name)
LANGS = {
    "eng": "eng_Latn",   # control / reference language
    "hin": "hin_Deva",
    "ben": "ben_Beng",
    "mar": "mar_Deva",
    "kan": "kan_Knda",   # Dravidian
    "tam": "tam_Taml",   # Dravidian
    "tel": "tel_Telu",   # Dravidian
}


def norm(line: str) -> str:
    """The ONLY preprocessing we apply. Documented in A1.

    - strip trailing newline / surrounding whitespace
    - NFC normalisation (see AUDIT.md F6: measured no-op on this data,
      but it makes the byte and grapheme denominators well-defined)
    - collapse nothing, lowercase nothing, strip no punctuation:
      we want to measure the text as it would actually be served.
    """
    return unicodedata.normalize("NFC", line.strip())


def fetch_flores():
    os.makedirs(CACHE, exist_ok=True)
    tgz = os.path.join(CACHE, "flores200_dataset.tar.gz")
    if not os.path.exists(tgz):
        print(f"  downloading {FLORES_URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(FLORES_URL, tgz)
    out = {}
    with tarfile.open(tgz) as tf:
        for short, code in LANGS.items():
            member = f"./flores200_dataset/devtest/{code}.devtest"
            f = tf.extractfile(member)
            if f is None:
                raise SystemExit(f"missing {member} in tarball")
            out[short] = [norm(l) for l in io.TextIOWrapper(f, encoding="utf-8")]
    return out


def fetch_in22():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("mteb/IN22-Gen", "test.parquet",
                        repo_type="dataset", cache_dir=CACHE)
    t = pq.read_table(p)
    return {short: [norm(s) for s in t.column(code).to_pylist()]
            for short, code in LANGS.items()}


def write(name, data):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    # Parallel-alignment guard: drop any index where ANY language is empty,
    # so line i means the same thing in every file.  Report how many.
    n = len(next(iter(data.values())))
    assert all(len(v) == n for v in data.values()), "corpora not line-aligned"
    keep = [i for i in range(n) if all(data[l][i] for l in data)]
    dropped = n - len(keep)
    for lang, lines in data.items():
        with open(os.path.join(d, f"{lang}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines[i] for i in keep) + "\n")
    meta = {
        "corpus": name,
        "languages": {k: LANGS[k] for k in data},
        "raw_lines": n,
        "dropped_blank_or_unaligned": dropped,
        "sentences": len(keep),
        "sha256": {
            lang: hashlib.sha256(
                open(os.path.join(d, f"{lang}.txt"), "rb").read()).hexdigest()[:16]
            for lang in data
        },
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: {len(keep)} parallel sentences x {len(data)} langs "
          f"({dropped} dropped)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        for n in ("flores", "in22"):
            print(json.dumps(json.load(
                open(os.path.join(OUT, n, "manifest.json"))), indent=2))
        return
    print("building corpus ...")
    write("flores", fetch_flores())
    write("in22", fetch_in22())


if __name__ == "__main__":
    main()
