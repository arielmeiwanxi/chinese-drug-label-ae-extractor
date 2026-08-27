"""Extract adverse reactions from the tabulated list of a Chinese label (LPI).

The Chinese label's AE table is laid out exactly like the EU SmPC's: MedDRA
system organ class, frequency, then the terms. So the same structural
extraction works, and the tabulated portion needs no LLM at all.

What the Chinese side does need is a Chinese-to-English mapping, because UMLS
carries no Chinese MedDRA. That comes from the curated dictionary
(LPI/ae_term_dictionary.csv). Terms the dictionary doesn't cover are reported
as unmapped rather than dropped: an unmapped term is a real gap that a human
has to close once, after which the dictionary covers it on every later run.

Usage:
  python extract_lpi_ae.py <label.pdf> <meddra_index.parquet> <out.csv>
"""
import csv
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).parent.parent
DICTIONARY = ROOT / "LPI" / "ae_term_dictionary.csv"

AE_HEADING = "【不良反应】"
NEXT_HEADING = "【禁忌】"
FOOTNOTE = re.compile(r"[*†#]+")
# "注射部位反应（包括红斑、水肿、瘙痒、疼痛、肿胀和淤青）" is one table row but
# six codable concepts; the prefix distributes over the parenthesised list.
COMPOUND = re.compile(r"^(.*?)反应\s*[（(]\s*包括(.+?)\s*[）)]\s*$")


def load_dictionary():
    rows = {}
    if DICTIONARY.exists():
        with open(DICTIONARY, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows[r["Chinese Term"].strip()] = r["Standard English Term"].strip()
    return rows


def load_lookup(index_path):
    import pandas as pd

    lut = {}
    for r in pd.read_parquet(index_path).itertuples():
        lut.setdefault(r.term_lower, []).append((r.tty, r.term, r.pt_term))
    return lut


def to_pt(term, lut):
    hits = lut.get(term.lower().strip())
    if not hits:
        return None, "not in MedDRA"
    pt = next((h for h in hits if h[0] == "PT"), None)
    if pt:
        return pt[1], ""
    _, name, roll = hits[0]
    return roll, f"coded as LLT '{name}'; PT is '{roll}'"


def ae_pages(pdf):
    """Pages between the 【不良反应】 heading and the next section heading."""
    start = end = None
    for i, p in enumerate(pdf.pages):
        t = p.extract_text() or ""
        if start is None and AE_HEADING in t:
            start = i
        elif start is not None and NEXT_HEADING in t:
            end = i
            break
    return range(start, (end if end is not None else start + 6) + 1) if start is not None else range(0)


def cell_lines(page, bbox):
    if not bbox:
        return []
    x0, top, x1, bottom = bbox
    crop = page.crop((x0, top, x1, min(bottom, page.height)))
    return [(l["top"], l["text"].strip()) for l in crop.extract_text_lines() if l["text"].strip()]


def extract_table(page):
    """A cell can hold several terms while the frequency cell beside it holds
    fewer, so terms are matched to frequencies by vertical position."""
    tables = page.find_tables()
    if not tables:
        return []
    out = []
    for row in tables[0].rows:
        if len(row.cells) < 3:
            continue
        soc_c, freq_c, term_c = row.cells[:3]
        soc = "".join(t for _, t in cell_lines(page, soc_c))
        if "系统器官分类" in soc:      # header row
            continue
        freqs = cell_lines(page, freq_c)
        for y, txt in cell_lines(page, term_c):
            applicable = [f for fy, f in freqs if fy <= y + 2]
            out.append({
                "soc": soc,
                "frequency": applicable[-1] if applicable else "",
                "term": txt,
            })
    return out


def merge_wrapped(rows):
    """A parenthesised list can wrap across table lines; rejoin those before
    the compound term is decomposed."""
    merged = []
    for r in rows:
        if merged and merged[-1]["term"].count("（") > merged[-1]["term"].count("）"):
            merged[-1]["term"] += r["term"]
        else:
            merged.append(dict(r))
    return merged


def decompose(term):
    m = COMPOUND.match(term)
    if not m:
        return [term]
    prefix, items = m.group(1).strip(), m.group(2)
    parts = [p for p in re.split(r"[、,，]|和", items) if p.strip()]
    return [f"{prefix}{p.strip()}" for p in parts] or [term]


def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_lpi_ae.py <label.pdf> <meddra_index.parquet> <out.csv>")
        sys.exit(1)
    pdf_path, index_path, out_path = sys.argv[1:4]

    zh_to_en = load_dictionary()
    lut = load_lookup(index_path)

    raw = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = ae_pages(pdf)
        print(f"Adverse reactions section: pages {pages.start + 1}-{pages.stop}")
        for i in pages:
            raw += extract_table(pdf.pages[i])

    rows, unmapped = [], []
    seen = set()
    for r in merge_wrapped(raw):
        for zh in decompose(FOOTNOTE.sub("", r["term"]).strip()):
            if not zh or zh in seen:
                continue
            seen.add(zh)
            en = zh_to_en.get(zh)
            if not en:
                unmapped.append(zh)
                rows.append({
                    "Source": "Table", "SOC": r["soc"], "Frequency": r["frequency"],
                    "Chinese Term": zh, "Standard English Term": "", "PT": "",
                    "Finding": "no dictionary entry - needs one-time manual mapping",
                })
                continue
            pt, note = to_pt(en, lut)
            rows.append({
                "Source": "Table", "SOC": r["soc"], "Frequency": r["frequency"],
                "Chinese Term": zh, "Standard English Term": en, "PT": pt or "",
                "Finding": note if pt else "not in MedDRA",
            })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"-> {out_path}")
    print(f"   {len(rows)} terms from the tabulated list")
    print(f"   {len(rows) - len(unmapped)} mapped to a PT, {len(unmapped)} unmapped")
    for u in unmapped:
        print(f"     · {u}")


if __name__ == "__main__":
    main()
