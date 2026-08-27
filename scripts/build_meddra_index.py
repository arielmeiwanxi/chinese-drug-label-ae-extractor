"""Build a MedDRA PT/LLT lookup table from UMLS MRCONSO.RRF.

Two things this gives you that a plain term list does not:

1. **LLT -> PT roll-up.** On a MedDRA LLT row, MRCONSO's SDUI column holds the
   code of the LLT's Preferred Term. So mapping an LLT to its PT is a
   deterministic dictionary lookup — no fuzzy matching, no AI, no network.
   This is what makes synonym handling free: "Fever" and "Pyrexia" are
   different LLTs that roll up to the same PT (Pyrexia, 10037660), so terms
   that look nothing alike normalise to one code.

2. **Term-level detection.** Because every term carries its TTY, you can tell
   whether a term found in a document is a PT or an LLT. A reference document
   that is supposed to be coded at PT level but contains an LLT is a data
   quality finding, and the correct PT comes free via the same SDUI lookup.

NOTE ON LICENSING: the output of this script is derived from UMLS/MedDRA,
which are licensed datasets. Do not redistribute it. The script is safe to
share; the table it produces is not. It is gitignored for that reason.

Usage: python build_meddra_index.py <umls_dir> <output.parquet>
"""
import sys
from pathlib import Path

import pandas as pd

MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
    "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF",
]


# MedDRA covers far more than adverse reactions — investigations, procedures,
# social circumstances. Scanning prose for any MedDRA term therefore surfaces
# things like "adverse event", "overdose", "blood eosinophils" that are real
# terms but not reported reactions. UMLS semantic types separate the two.
AE_SEMANTIC_TYPES = {
    "T047",  # Disease or Syndrome
    "T184",  # Sign or Symptom
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T048",  # Mental or Behavioral Dysfunction
    "T019",  # Congenital Abnormality
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
}


def load_ae_cuis(mrsty_path):
    """CUIs whose semantic type makes them plausible adverse reactions."""
    cuis = set()
    with open(mrsty_path, encoding="utf-8") as f:
        for line in f:
            p = line.split("|", 2)
            if len(p) > 1 and p[1] in AE_SEMANTIC_TYPES:
                cuis.add(p[0])
    return cuis


def build(mrconso_path):
    rows = []
    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            if len(p) < 15 or p[11] != "MDR" or p[1] != "ENG":
                continue
            if p[16] not in ("N", "n", ""):  # SUPPRESS
                continue
            rows.append({
                "term": p[14],
                "term_lower": p[14].lower(),
                "tty": p[12],
                "meddra_code": p[13],
                "pt_code": p[10],   # SDUI — the Preferred Term's code
                "cui": p[0],
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["term_lower", "tty", "meddra_code"])


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_meddra_index.py <umls_dir> <output.parquet>")
        sys.exit(1)
    umls_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    mrconso = umls_dir / "MRCONSO.RRF"
    if not mrconso.exists():
        print(f"ERROR: {mrconso} not found.")
        sys.exit(1)

    print(f"Parsing {mrconso} (this reads ~2GB, takes a minute)...")
    df = build(mrconso)

    # Attach each row's PT name so a lookup returns the roll-up directly.
    pt_names = df[df["tty"] == "PT"].set_index("meddra_code")["term"].to_dict()
    df["pt_term"] = df["pt_code"].map(pt_names)

    mrsty = umls_dir / "MRSTY.RRF"
    if mrsty.exists():
        ae_cuis = load_ae_cuis(mrsty)
        df["is_ae_type"] = df["cui"].isin(ae_cuis)
        print(f"   {df['is_ae_type'].sum():,} terms carry an AE-plausible semantic type")
    else:
        df["is_ae_type"] = True
        print("   MRSTY.RRF not found — semantic-type filtering unavailable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_pt = (df["tty"] == "PT").sum()
    n_llt = (df["tty"] == "LLT").sum()
    resolved = df[df["tty"] == "LLT"]["pt_term"].notna().sum()
    print(f"-> {out_path}")
    print(f"   {len(df):,} MedDRA terms total")
    print(f"   {n_pt:,} PT / {n_llt:,} LLT")
    print(f"   {resolved:,} of {n_llt:,} LLTs ({resolved / n_llt * 100:.1f}%) resolve to a PT")


if __name__ == "__main__":
    main()
