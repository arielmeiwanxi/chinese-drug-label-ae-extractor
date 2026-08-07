"""Stage 5: resolve each candidate to a UMLS/MedDRA concept.

Two paths, matching how translate.py tagged each candidate's resolution_method:

- "dictionary": exact hit against the curated Chinese->English AE dictionary.
  The CUI came from a human-reviewed mapping, not a model guess, so this is
  taken as confirmed and does not need review.
- "nmt_fallback": the English text came from an unreliable local NMT model.
  Calibration against the real UMLS/MedDRA index showed mistranslations score
  just as high (0.90-1.00 cosine similarity) as correct translations do — the
  embedding score cannot tell "translated wrong, matched the wrong concept
  with high confidence" apart from "translated right, matched correctly". So
  NOTHING on this path is auto-accepted: every row is flagged for mandatory
  human review, and the score is used only to sort the review queue, never to
  promote a row to "confirmed".

Usage:
  python match_classify.py <candidates_translated.json> <umls_index.parquet> <report.csv>
"""
import argparse
import json

import numpy as np
import pandas as pd


def embed_candidates(phrases_en, model):
    return model.encode(phrases_en, normalize_embeddings=True, batch_size=64, show_progress_bar=True)


def best_matches(cand_emb, umls_df, umls_emb, top_k=3):
    sims = cand_emb @ umls_emb.T  # (n_candidates, n_umls_terms)
    top_idx = np.argsort(-sims, axis=1)[:, :top_k]
    results = []
    for i in range(sims.shape[0]):
        matches = [
            {
                "STR": umls_df.iloc[j]["STR"],
                "CUI": umls_df.iloc[j]["CUI"],
                "SAB": umls_df.iloc[j]["SAB"],
                "TTY": umls_df.iloc[j]["TTY"],
                "score": float(sims[i, j]),
            }
            for j in top_idx[i]
        ]
        results.append(matches)
    return results


def load_umls_index(path):
    df = pd.read_parquet(path)
    emb = np.vstack(df["embedding"].to_numpy())
    return df.reset_index(drop=True), emb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates_translated")
    parser.add_argument("umls_index")
    parser.add_argument("report_csv", nargs="?")
    parser.add_argument("--calibrate", help="ground-truth term CSV (column: 'Chinese Term', e.g. LPI/ae_term_dictionary.csv) to print score distribution instead of writing a report")
    args = parser.parse_args()

    with open(args.candidates_translated, encoding="utf-8") as f:
        candidates = json.load(f)

    dict_rows = [c for c in candidates if c.get("resolution_method") == "dictionary"]
    nmt_rows = [c for c in candidates if c.get("resolution_method") != "dictionary"]
    print(f"{len(dict_rows)} candidates resolved via dictionary (confirmed, no review needed)")
    print(f"{len(nmt_rows)} candidates need embedding fallback (all -> mandatory human review)")

    umls_df, umls_emb = load_umls_index(args.umls_index)

    if nmt_rows:
        from sentence_transformers import SentenceTransformer
        print("Loading biomedical embedding model...")
        model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
        phrases_en = [c["candidate_phrase_en"] for c in nmt_rows]
        print(f"Embedding {len(phrases_en)} NMT-fallback candidate phrases...")
        cand_emb = embed_candidates(phrases_en, model)
        print(f"Matching against {len(umls_df)} UMLS terms...")
        matches = best_matches(cand_emb, umls_df, umls_emb)
        for c, m in zip(nmt_rows, matches):
            c["top_matches"] = m
            c["best_score"] = m[0]["score"]

    if args.calibrate:
        gt = pd.read_csv(args.calibrate)
        gt_terms = set(gt["Chinese Term"])
        gt_scores = [c["best_score"] for c in nmt_rows if c["candidate_phrase"] in gt_terms]
        print(f"\n{len(gt_scores)} ground-truth terms went through NMT fallback path (rest resolved via dictionary)")
        if gt_scores:
            arr = np.array(gt_scores)
            print(f"  min={arr.min():.3f}  median={np.median(arr):.3f}  max={arr.max():.3f}")
        return

    STATUS_CONFIRMED = "Confirmed (dictionary)"
    STATUS_REVIEW = "Needs human review"

    rows = []
    for c in dict_rows:
        rows.append({
            "Section": c["section"],
            "Pages": f"{c['start_page']}-{c['end_page']}",
            "Chinese Term": c["candidate_phrase"],
            "Context": c["context_sentence"],
            "Resolution Method": "Dictionary exact match",
            "Translated English": c["candidate_phrase_en"],
            "Matched UMLS Term": c["candidate_phrase_en"],
            "CUI": c["dict_cui"],
            "SAB": c["dict_sab"],
            "TTY": c["dict_tty"],
            "Similarity Score": 1.0,
            "Status": STATUS_CONFIRMED,
            "Alternative Term 2": "",
            "Alternative Term 3": "",
        })
    for c in nmt_rows:
        top = c["top_matches"][0]
        rows.append({
            "Section": c["section"],
            "Pages": f"{c['start_page']}-{c['end_page']}",
            "Chinese Term": c["candidate_phrase"],
            "Context": c["context_sentence"],
            "Resolution Method": "NMT + embedding fallback",
            "Translated English": c["candidate_phrase_en"],
            "Matched UMLS Term": top["STR"],
            "CUI": top["CUI"],
            "SAB": top["SAB"],
            "TTY": top["TTY"],
            "Similarity Score": round(top["score"], 4),
            "Status": STATUS_REVIEW,
            "Alternative Term 2": c["top_matches"][1]["STR"] if len(c["top_matches"]) > 1 else "",
            "Alternative Term 3": c["top_matches"][2]["STR"] if len(c["top_matches"]) > 2 else "",
        })

    out_df = pd.DataFrame(rows)
    # sort so the human-review queue is prioritized by score, dictionary rows first
    out_df["_sort"] = out_df["Status"].map({STATUS_CONFIRMED: 0, STATUS_REVIEW: 1})
    out_df = out_df.sort_values(["_sort", "Similarity Score"], ascending=[True, False]).drop(columns="_sort")
    out_df.to_csv(args.report_csv, index=False, encoding="utf-8-sig")
    n_confirmed = (out_df["Status"] == STATUS_CONFIRMED).sum()
    n_review = (out_df["Status"] == STATUS_REVIEW).sum()
    print(f"-> {args.report_csv} ({n_confirmed} confirmed via dictionary, {n_review} need human review)")


if __name__ == "__main__":
    main()
