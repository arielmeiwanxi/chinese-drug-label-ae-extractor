"""Orchestrates the full pipeline end to end.

Pipeline:
  1. extract_sections.py  - PDF -> section-tagged text (layout-aware, no LLM/API)
  2. llm_extract.py       - LLM reads each section, proposes AE terms (needs API key)
  3. build_umls_index.py  - parse UMLS RRF files -> embedding index (run once, cached)
  4. verify_and_report.py - deterministically verify every proposed term against
                             real UMLS/MedDRA data (exact lookup, not similarity score)

Usage: python run_pipeline.py
Assumes the standard project layout:
  LPI/*.pdf
  umls/MRCONSO.RRF (+ optional MRSTY.RRF)  -- see README for where to get these
  output/  (all intermediate + final files land here)

Requires an LLM API key (see llm_extract.py docstring for provider setup).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LPI_DIR = ROOT / "LPI"
UMLS_DIR = ROOT / "umls"
OUT_DIR = ROOT / "output"
SCRIPTS = Path(__file__).parent


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([sys.executable, *cmd], check=True, cwd=ROOT)


def main():
    pdfs = list(LPI_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF found in {LPI_DIR}")
        sys.exit(1)
    pdf_path = pdfs[0]

    OUT_DIR.mkdir(exist_ok=True)

    run([SCRIPTS / "extract_sections.py", pdf_path, OUT_DIR / "sections.json"])
    run([SCRIPTS / "llm_extract.py", OUT_DIR / "sections.json", OUT_DIR / "llm_candidates.json"])

    umls_index = OUT_DIR / "umls_meddra_index.parquet"
    if not umls_index.exists():
        run([SCRIPTS / "build_umls_index.py", UMLS_DIR, umls_index])
    else:
        print(f"\nReusing existing UMLS index: {umls_index}")

    run([SCRIPTS / "verify_and_report.py", OUT_DIR / "llm_candidates.json", umls_index, OUT_DIR / "ae_report.csv"])

    print(f"\nDone. Report: {OUT_DIR / 'ae_report.csv'}")


if __name__ == "__main__":
    main()
