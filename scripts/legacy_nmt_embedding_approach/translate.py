"""Stage 3: resolve each candidate phrase to English.

Dictionary-first: exact matches against the curated Chinese->English AE term
dictionary (LPI/ae_term_dictionary.csv) are used as-is — they carry a CUI
directly and don't need NMT or embedding matching at all. Everything else
falls back to local NMT, but NMT output is NOT trustworthy on its own (see
calibration results: mistranslations score just as high as correct ones on
embedding similarity) — so anything on this path is downstream forced into
mandatory human review regardless of its match score, never auto-accepted.

Usage: python translate.py <candidates.json> <candidates_translated.json>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / "output" / "translation_cache.json"
DICTIONARY_PATH = ROOT / "LPI" / "ae_term_dictionary.csv"
MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"


def load_dictionary():
    if not DICTIONARY_PATH.exists():
        return {}
    df = pd.read_csv(DICTIONARY_PATH)
    return {
        row["Chinese Term"]: {"en": row["Standard English Term"], "CUI": row["CUI"], "SAB": row["SAB"]}
        for _, row in df.iterrows()
    }


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_batch(texts, cache, tokenizer, model, batch_size=32):
    import torch

    to_translate = [t for t in texts if t not in cache]
    for i in range(0, len(to_translate), batch_size):
        batch = to_translate[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for src, tgt in zip(batch, translations):
            cache[src] = tgt
        print(f"  NMT fallback translated {min(i + batch_size, len(to_translate))}/{len(to_translate)}")
    return cache


def main():
    if len(sys.argv) != 3:
        print("Usage: python translate.py <candidates.json> <candidates_translated.json>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, encoding="utf-8") as f:
        candidates = json.load(f)

    dictionary = load_dictionary()
    print(f"{len(dictionary)} terms in curated dictionary ({DICTIONARY_PATH.name})")

    dict_hits = [c for c in candidates if c["candidate_phrase"] in dictionary]
    nmt_needed = [c for c in candidates if c["candidate_phrase"] not in dictionary]
    print(f"{len(dict_hits)} candidates resolved via dictionary (exact match)")
    print(f"{len(nmt_needed)} candidates need NMT fallback (will require mandatory human review downstream)")

    unique_phrases = sorted({c["candidate_phrase"] for c in nmt_needed})
    cache = load_cache()
    if unique_phrases:
        print("Loading NMT fallback model (first run downloads weights)...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        cache = translate_batch(unique_phrases, cache, tokenizer, model)
        save_cache(cache)

    for c in candidates:
        phrase = c["candidate_phrase"]
        if phrase in dictionary:
            d = dictionary[phrase]
            c["candidate_phrase_en"] = d["en"]
            c["resolution_method"] = "dictionary"
            c["dict_cui"] = d["CUI"]
            c["dict_sab"] = d["SAB"]
            c["dict_tty"] = ""  # the dictionary CSV doesn't carry a TTY column
        else:
            c["candidate_phrase_en"] = cache[phrase]
            c["resolution_method"] = "nmt_fallback"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
