"""Run the whole reconciliation end to end.

    python scripts/run_pipeline.py

Expects the standard layout:
    umls/MRCONSO.RRF, umls/MRSTY.RRF   licensed UMLS release, see README
    SmPC/*.pdf                          reference label (EU SmPC)
    LPI/*.pdf                           local label (Chinese PI)

Everything lands in output/. The MedDRA index is the expensive step (it reads
a ~2GB file) so it is built once and reused; delete it to force a rebuild.

No LLM and no network are used by this path. The narrative sections of the
local label are not yet extracted automatically, which compare_labels.py
detects and warns about rather than reporting as differences.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
OUT = ROOT / "output"


def run(script, *args):
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"\n$ python scripts/{script} {' '.join(map(str, args))}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def only_pdf(folder):
    pdfs = sorted((ROOT / folder).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDF found in {folder}/. See the README for what belongs there.")
    return pdfs[0]


def main():
    OUT.mkdir(exist_ok=True)
    index = OUT / "meddra_index.parquet"

    if index.exists():
        print(f"Reusing MedDRA index: {index}")
    else:
        run("build_meddra_index.py", ROOT / "umls", index)

    run("extract_smpc_ae.py", only_pdf("SmPC"), index, OUT / "smpc_ae.csv")
    run("extract_lpi_ae.py", only_pdf("LPI"), index, OUT / "lpi_ae.csv")
    run("compare_labels.py", OUT / "smpc_ae.csv", OUT / "lpi_ae.csv",
        index, OUT / "label_diff.csv")

    print(f"\nDone. Reconciliation report: {OUT / 'label_diff.csv'}")


if __name__ == "__main__":
    main()
