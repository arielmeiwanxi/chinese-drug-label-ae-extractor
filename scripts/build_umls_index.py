"""Stage 4: parse UMLS MRCONSO.RRF, filter to relevant English terms, embed
every term string with a local biomedical sentence-embedding model, and
cache the result (strings + CUI/SAB/TTY metadata + embeddings) to parquet
so later runs don't have to recompute it.

Expects umls/MRCONSO.RRF (required) and optionally umls/MRSTY.RRF (semantic
type filtering) under the project root.

Usage: python build_umls_index.py <umls_dir> <output.parquet> [--all-vocab]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
    "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF",
]

# MedDRA is the pharmacovigilance standard and matches how this LPI itself
# is organized (by MedDRA System Organ Class). Restrict to MedDRA by default;
# pass --all-vocab to widen to the full English UMLS instead.
MEDDRA_SABS = {"MDR"}

# AE-relevant semantic types, used only if MRSTY.RRF is present.
AE_SEMANTIC_TYPES = {
    "T047",  # Disease or Syndrome
    "T184",  # Sign or Symptom
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T048",  # Mental or Behavioral Dysfunction
    "T033",  # Finding
    "T019",  # Congenital Abnormality
    "T190",  # Anatomical Abnormality
}


def load_mrconso(path, use_all_vocab):
    df = pd.read_csv(
        path, sep="|", header=None, names=MRCONSO_COLS + ["_"],
        dtype=str, index_col=False, quoting=3,
    )
    df = df[df["LAT"] == "ENG"]
    if not use_all_vocab:
        df = df[df["SAB"].isin(MEDDRA_SABS)]
    df = df[df["SUPPRESS"].isin(["N", "n"]) | df["SUPPRESS"].isna()]
    df = df[["CUI", "SAB", "TTY", "CODE", "STR", "ISPREF"]].drop_duplicates(subset=["CUI", "STR"])
    return df.reset_index(drop=True)


def filter_by_semantic_type(df, mrsty_path):
    sty = pd.read_csv(
        mrsty_path, sep="|", header=None,
        names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "_"],
        dtype=str, index_col=False, quoting=3,
    )
    ae_cuis = set(sty.loc[sty["TUI"].isin(AE_SEMANTIC_TYPES), "CUI"])
    return df[df["CUI"].isin(ae_cuis)].reset_index(drop=True)


def main():
    if len(sys.argv) < 3:
        print("Usage: python build_umls_index.py <umls_dir> <output.parquet> [--all-vocab]")
        sys.exit(1)
    umls_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    use_all_vocab = "--all-vocab" in sys.argv

    mrconso_path = umls_dir / "MRCONSO.RRF"
    if not mrconso_path.exists():
        print(f"ERROR: {mrconso_path} not found. Place MRCONSO.RRF under {umls_dir}/")
        sys.exit(1)

    print(f"Parsing {mrconso_path} ...")
    df = load_mrconso(mrconso_path, use_all_vocab)
    print(f"  {len(df)} English terms after SAB filter ({'ALL UMLS' if use_all_vocab else 'MedDRA only'})")

    mrsty_path = umls_dir / "MRSTY.RRF"
    if mrsty_path.exists():
        print(f"Filtering by AE-relevant semantic types via {mrsty_path} ...")
        df = filter_by_semantic_type(df, mrsty_path)
        print(f"  {len(df)} terms remain after semantic-type filter")
    else:
        print("  MRSTY.RRF not found, skipping semantic-type filter")

    print("Loading biomedical sentence embedding model (pritamdeka/S-PubMedBert-MS-MARCO)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

    print(f"Embedding {len(df)} terms...")
    embeddings = model.encode(
        df["STR"].tolist(), normalize_embeddings=True,
        batch_size=256, show_progress_bar=True,
    )

    df["embedding"] = list(embeddings.astype(np.float32))
    df.to_parquet(out_path)
    print(f"-> {out_path} ({len(df)} rows, {embeddings.shape[1]}-dim embeddings)")


if __name__ == "__main__":
    main()
