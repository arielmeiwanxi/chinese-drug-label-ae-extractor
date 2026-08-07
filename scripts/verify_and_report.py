"""Stage 4: deterministically verify every LLM-proposed English term against
the real UMLS/MedDRA data (build_umls_index.py's output) and produce the
final report.

This is a lookup, not a similarity score: an LLM proposing "Conjunctivitis"
either finds an exact MedDRA string match (verified) or it doesn't (flagged).
There is no threshold to miscalibrate and no way for a wrong term to sneak
through with a deceptively high score — a term either exists in MedDRA under
(close to) that exact string, or it's flagged for a human to code manually.

Dictionary hits (scripts/../LPI/ae_term_dictionary.csv) skip the LLM
entirely — they're already human-confirmed from a prior run, so re-spending
an API call and re-verifying is pure waste.

Output columns are English (this is a portfolio/general-purpose repo), except
for two columns that hold text verbatim from the source PDF and must stay in
the original language for anyone to be able to trace a result back to its
source and verify it: Chinese Term, and Context.

Usage: python verify_and_report.py <llm_candidates.json> <umls_index.parquet> <report.csv>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DICTIONARY_PATH = ROOT / "LPI" / "ae_term_dictionary.csv"

STATUS_DICT_HIT = "Dictionary hit (pre-verified, no LLM call needed)"
STATUS_EXACT = "UMLS verified (exact match)"
STATUS_SPELLING = "UMLS verified (British/American spelling variant)"
STATUS_NOT_FOUND = "Not found in UMLS - needs manual coding"


def load_dictionary():
    if not DICTIONARY_PATH.exists():
        return {}
    df = pd.read_csv(DICTIONARY_PATH)
    return {row["Chinese Term"]: row for _, row in df.iterrows()}


def normalize(s):
    # British/American spelling is the single most common reason an
    # otherwise-correct MedDRA term string fails an exact match.
    return (
        s.strip().lower()
        .replace("oedema", "edema")
        .replace("haemorrhage", "hemorrhage")
        .replace("anaemia", "anemia")
    )


def verify_term(en_term, umls_df, norm_lookup):
    exact = umls_df[umls_df["STR"].str.lower() == en_term.strip().lower()]
    if len(exact):
        row = exact[exact["TTY"] == "PT"]
        row = row.iloc[0] if len(row) else exact.iloc[0]
        return STATUS_EXACT, row["STR"], row["CUI"], row["TTY"]

    norm = normalize(en_term)
    if norm in norm_lookup:
        row = norm_lookup[norm]
        return STATUS_SPELLING, row["STR"], row["CUI"], row["TTY"]

    return STATUS_NOT_FOUND, "", "", ""


def suggest_nearest(unverified_terms, umls_df):
    """For terms with no exact/normalized match, suggest nearby MedDRA terms
    by embedding similarity — advisory only, for a human doing manual coding
    to consider. Never used to auto-classify (see legacy_nmt_embedding_approach/
    README.md for why a similarity score alone can't be trusted for that)."""
    if not unverified_terms or "embedding" not in umls_df.columns:
        return {}
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"\nComputing embedding suggestions for {len(unverified_terms)} unverified term(s) (advisory only)...")
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    umls_emb = np.vstack(umls_df["embedding"].to_numpy()).astype(np.float32)
    term_emb = model.encode(unverified_terms, normalize_embeddings=True)

    sims = term_emb @ umls_emb.T
    top_idx = np.argsort(-sims, axis=1)[:, :3]
    suggestions = {}
    for i, term in enumerate(unverified_terms):
        picks = [f"{umls_df.iloc[j]['STR']} ({sims[i, j]:.2f})" for j in top_idx[i]]
        suggestions[term] = "; ".join(picks)
    return suggestions


def main():
    if len(sys.argv) != 4:
        print("Usage: python verify_and_report.py <llm_candidates.json> <umls_index.parquet> <report.csv>")
        sys.exit(1)
    import json
    candidates_path, umls_index_path, report_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(candidates_path, encoding="utf-8") as f:
        llm_candidates = json.load(f)

    dictionary = load_dictionary()
    import pyarrow.parquet as pq
    # schema_arrow (not schema/schema.names, which flattens nested list
    # columns down to their inner element field names) — metadata only, no data read
    parquet_cols = pq.ParquetFile(umls_index_path).schema_arrow.names
    cols = ["CUI", "SAB", "TTY", "CODE", "STR"] + (["embedding"] if "embedding" in parquet_cols else [])
    umls_df = pd.read_parquet(umls_index_path, columns=cols)
    norm_lookup = {}
    for _, row in umls_df.iterrows():
        key = normalize(row["STR"])
        if key not in norm_lookup:
            norm_lookup[key] = row

    rows = []
    seen_zh = set()
    for c in llm_candidates:
        zh = c["zh"].strip()
        if zh in seen_zh:
            continue
        seen_zh.add(zh)

        if zh in dictionary:
            d = dictionary[zh]
            rows.append({
                "Section": c["section"], "Pages": f"{c['start_page']}-{c['end_page']}",
                "Chinese Term": zh, "Context": c.get("context", ""),
                "Standard English Term": d["Standard English Term"], "CUI": d["CUI"],
                "SAB": d.get("SAB", "MDR"), "TTY": "",
                "Status": STATUS_DICT_HIT, "LLM Confidence": "", "Note": "", "_llm_en": "",
            })
            continue

        status, matched_str, cui, tty = verify_term(c["en"], umls_df, norm_lookup)
        rows.append({
            "Section": c["section"], "Pages": f"{c['start_page']}-{c['end_page']}",
            "Chinese Term": zh, "Context": c.get("context", ""),
            "Standard English Term": matched_str or c["en"], "CUI": cui,
            "SAB": "MDR" if cui else "", "TTY": tty,
            "Status": status, "LLM Confidence": c.get("confidence", ""), "Note": c.get("note", ""),
            "_llm_en": c["en"],
        })

    unverified_terms = sorted({r["_llm_en"] for r in rows if r["Status"] == STATUS_NOT_FOUND})
    suggestions = suggest_nearest(unverified_terms, umls_df)
    for r in rows:
        r["Suggested Alternatives (advisory only, not auto-classified)"] = suggestions.get(r["_llm_en"], "")
        del r["_llm_en"]

    df = pd.DataFrame(rows)
    status_order = {STATUS_DICT_HIT: 0, STATUS_EXACT: 1, STATUS_SPELLING: 2, STATUS_NOT_FOUND: 3}
    df["_sort"] = df["Status"].map(status_order)
    df = df.sort_values("_sort").drop(columns="_sort")
    df.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\n{len(df)} unique AE candidate(s) -> {report_path}")
    print(df["Status"].value_counts().to_string())


if __name__ == "__main__":
    main()
