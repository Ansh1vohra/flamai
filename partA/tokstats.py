#!/usr/bin/env python3
"""
tokstats.py -- shared measurement core for Part A.

Everything downstream (the audit ablations and the corrected analysis)
imports from here so that there is exactly one definition of each
denominator and one aggregation rule.
"""
import functools, os, unicodedata
import regex  # for \X, the Unicode extended grapheme cluster

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- tokenizers
# Two "production-shaped" decoder tokenizers + two Indic-aware ones.
TOKENIZERS = {
    # what the intern used; OpenAI GPT-2 byte-level BPE, 50k vocab, English-only training
    "gpt2":       ("tiktoken", "gpt2"),
    # what a current OpenAI-class model actually serves with, 100k vocab
    "cl100k":     ("tiktoken", "cl100k_base"),
    # Indic-first *decoder* LLM tokenizer (Sarvam-1, 68k vocab, 10 Indian languages)
    "sarvam-1":   ("hf", "sarvamai/sarvam-1"),
    # Indic-specialised encoder tokenizer (Google MuRIL, 197k vocab, 17 Indian languages)
    "muril":      ("hf", "google/muril-base-cased"),
}


@functools.lru_cache(maxsize=None)
def get_encoder(name: str):
    kind, spec = TOKENIZERS[name]
    if kind == "tiktoken":
        import tiktoken
        return tiktoken.get_encoding(spec).encode
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec)
    # add_special_tokens=False is deliberate -- see AUDIT.md F7.
    return lambda s: tok.encode(s, add_special_tokens=False)


# --------------------------------------------------------------- denominators
# Each returns the count of the unit for one line.
def n_words(s):       return len(s.split())                    # whitespace words
def n_codepoints(s):  return len(s)                            # what len() gives you
def n_graphemes(s):   return len(regex.findall(r"\X", s))      # user-perceived chars
def n_bytes(s):       return len(s.encode("utf-8"))            # UTF-8 bytes
def n_sentences(s):   return 1                                 # parallel-sentence unit

DENOMS = {
    "word":      n_words,
    "grapheme":  n_graphemes,
    "byte":      n_bytes,
    "codepoint": n_codepoints,
    "sentence":  n_sentences,
}


# ------------------------------------------------------------------ i/o + agg
def read_lines(path, normalize=True):
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if normalize:
                line = unicodedata.normalize("NFC", line)
            out.append(line)
    return out


def totals(lines, encode, denoms=DENOMS):
    """Corpus-level sums. Micro-averaging (sum tokens / sum units) is the
    correct aggregation for a cost question: you pay for the corpus, not
    for the average of per-sentence rates.  See AUDIT.md F3."""
    tok = 0
    units = {k: 0 for k in denoms}
    for line in lines:
        tok += len(encode(line))
        for k, fn in denoms.items():
            units[k] += fn(line)
    return tok, units


def micro(lines, encode, denoms=DENOMS):
    tok, units = totals(lines, encode, denoms)
    return {k: tok / units[k] for k in denoms}, tok, units


def macro(lines, encode, denoms=DENOMS):
    """Mean of per-line ratios -- what fertility.py v0 does. Kept so the
    audit can quantify the difference rather than assert it."""
    acc = {k: 0.0 for k in denoms}
    for line in lines:
        t = len(encode(line))
        for k, fn in denoms.items():
            acc[k] += t / fn(line)
    return {k: v / len(lines) for k, v in acc.items()}


def corpus_path(corpus, lang):
    return os.path.join(HERE, "corpus", corpus, f"{lang}.txt")


LANGS = ["eng", "hin", "ben", "mar", "kan", "tam", "tel"]
