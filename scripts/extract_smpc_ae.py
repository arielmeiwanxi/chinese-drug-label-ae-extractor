"""Extract adverse reactions from an EMA SmPC (Section 4.8) — no LLM required.

The SmPC is English and its AEs are already MedDRA-ish, so the whole document
side can be handled by deterministic dictionary lookup against the MedDRA
index. That matters twice over: it costs nothing to run, and it means a
confidential English reference document (a CCDS, in the real-world version of
this task) never has to leave the machine.

Three things make the free-text scan usable rather than noisy:

1. **Longest-match-first with overlap suppression.** Without it, the LLT
   "Sickness" matches inside "serum sickness" and drags in its unrelated PT
   ("Illness"). Candidates are tried longest-first and a match is dropped if
   it overlaps a span already claimed by a longer term.

2. **Indication exclusion.** The diseases the drug *treats* are all over
   Section 4.8's prose ("...in atopic dermatitis studies..."), and they are
   valid MedDRA terms, so a naive scan reports them as adverse reactions.
   The exclusion list is read from the document's own Section 4.1 rather
   than hardcoded, so it stays correct for a different drug.

3. **Subheadings ranked above prose.** In "Description of selected adverse
   reactions" the subheadings *are* the AEs, so they are extracted separately
   and marked high-confidence; prose matches are marked for review.

Usage: python extract_smpc_ae.py <smpc.pdf> <meddra_index.parquet> <out.csv>
"""
import csv
import re
import sys

import pdfplumber

SENTENCE_END = re.compile(r"[.;:]\s*$")


def find_smpc_range(pdf):
    """The EMA Product Information bundles several SmPCs (one per strength)
    plus the annexes. Return the page range of the first SmPC only."""
    starts = [
        i for i, p in enumerate(pdf.pages)
        if re.search(r"^\s*1\.\s+NAME OF THE MEDICINAL PRODUCT", p.extract_text() or "", re.M)
    ]
    end = next(
        (i for i, p in enumerate(pdf.pages) if re.search(r"^ANNEX II", p.extract_text() or "", re.M)),
        len(pdf.pages),
    )
    first = starts[0] if starts else 0
    second = next((s for s in starts if s > first), end)
    return first, min(second, end)


def section_pages(pdf, lo, hi, pattern):
    return [i for i in range(lo, hi) if re.search(pattern, pdf.pages[i].extract_text() or "")]


def extract_indications(pdf, lo, hi):
    """Section 4.1 lists each indication as a subheading. Those are the
    diseases the drug treats — never adverse reactions."""
    txt = "\n".join((pdf.pages[i].extract_text() or "") for i in range(lo, min(lo + 10, hi)))
    s, e = txt.find("4.1 Therapeutic indications"), txt.find("4.2 Posology")
    if s < 0:
        return set()
    body = txt[s:e] if e > s else txt[s:]
    out = set()
    for line in body.split("\n")[1:]:
        line = line.strip()
        if not line or len(line) > 60 or SENTENCE_END.search(line):
            continue
        if line.startswith("Dupixent") or re.match(r"^(Adults|Children|Adolescents)", line):
            continue
        out.add(re.sub(r"\s*\([A-Za-z]+\)$", "", line).strip().lower())
    return out


def cell_lines(page, bbox):
    if not bbox:
        return []
    x0, top, x1, bottom = bbox
    crop = page.crop((x0, top, x1, min(bottom, page.height)))
    return [(l["top"], l["text"].strip()) for l in crop.extract_text_lines() if l["text"].strip()]


def extract_table(page):
    """Parse the tabulated AE list. A cell can hold several terms while the
    frequency cell beside it holds fewer, so terms are matched to frequencies
    by vertical position rather than by row order."""
    tables = page.find_tables()
    if not tables:
        return []
    tbl = tables[0]
    rows = []
    for r in tbl.rows[1:]:
        if len(r.cells) < 3:
            continue
        soc_c, freq_c, term_c = r.cells[:3]
        soc = " ".join(t for _, t in cell_lines(page, soc_c))
        freqs = cell_lines(page, freq_c)
        pending = []
        for y, txt in cell_lines(page, term_c):
            # A line starting lowercase is a wrap of the line above it.
            if pending and txt[:1].islower():
                pending[-1] = (pending[-1][0], pending[-1][1] + " " + txt)
            else:
                pending.append((y, txt))
        for y, txt in pending:
            applicable = [f for fy, f in freqs if fy <= y + 2]
            rows.append({
                "soc": re.sub(r"\s+", " ", soc),
                "frequency": applicable[-1] if applicable else "",
                "term": re.sub(r"[*†#]+", "", txt).strip(),
            })
    return rows


def decompose(term):
    """"Injection site reactions (includes erythema, oedema, pruritus, ...)"
    is one table row but six codable concepts. Expand it."""
    m = re.match(r"^(.*?)\s+reactions?\s*\((?:includes?|including)\s+(.+)\)\s*$", term, re.I)
    if not m:
        return [term]
    prefix, items = m.group(1).strip(), m.group(2)
    parts = [p.strip() for p in re.split(r",|\band\b", items) if p.strip()]
    return [f"{prefix} {p}" for p in parts] or [term]


def build_lookup(index_path):
    import pandas as pd

    idx = pd.read_parquet(index_path)
    lut, ae_only = {}, set()
    for r in idx.itertuples():
        lut.setdefault(r.term_lower, []).append((r.tty, r.term, r.pt_term))
        if getattr(r, "is_ae_type", True):
            ae_only.add(r.term_lower)
    return lut, ae_only


def resolve(term, lut):
    """Return (pt, tty, note). A term that exists only as an LLT is a coding
    finding: the document used a lower-level term where a PT was expected,
    and the correct PT comes back with it."""
    hits = lut.get(term.lower().strip())
    if not hits:
        return None, None, "not in MedDRA - needs manual coding"
    pt_hit = next((h for h in hits if h[0] == "PT"), None)
    if pt_hit:
        return pt_hit[1], "PT", ""
    tty, name, pt = hits[0]
    return pt, "LLT", f"coded as LLT '{name}'; PT is '{pt}'"


def narrative_subheadings(text):
    """In "Description of selected adverse reactions" each subheading names
    the reaction the paragraph below it describes. That structure is a far
    better signal than scanning the prose, so these are reported separately
    and at high confidence."""
    marker = "Description of selected adverse reactions"
    if marker not in text:
        return []
    body = text[text.find(marker) + len(marker):]
    out = []
    for line in body.split("\n"):
        line = line.strip()
        if not (0 < len(line) < 50) or not line[:1].isupper():
            continue
        # Real subheadings are bare noun phrases: no sentence punctuation,
        # no digits, no study-code parentheticals.
        if SENTENCE_END.search(line) or re.search(r"[\d;()%]", line):
            continue
        if line not in out:
            out.append(line)
    return out


def scan_narrative(text, lut, exclude, ae_only):
    """Longest-match-first scan with overlap suppression.

    Every occurrence of a term is claimed, not just the first — otherwise a
    short term slips inside the *second* occurrence of a longer one, which is
    how "sickness" ends up matching inside "serum sickness" and reporting the
    unrelated PT "Illness"."""
    low = text.lower()
    terms = sorted(
        (t for t in lut if (len(t) >= 8 or " " in t) and t not in exclude and t in ae_only),
        key=len, reverse=True,
    )
    claimed, found = [], {}
    for t in terms:
        for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
            if any(m.start() < e and m.end() > s for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            found.setdefault(t, text[max(0, m.start() - 60):m.end() + 60].replace("\n", " "))
    return found


def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_smpc_ae.py <smpc.pdf> <meddra_index.parquet> <out.csv>")
        sys.exit(1)
    pdf_path, index_path, out_path = sys.argv[1:4]

    lut, ae_only = build_lookup(index_path)
    with pdfplumber.open(pdf_path) as pdf:
        lo, hi = find_smpc_range(pdf)
        print(f"SmPC pages {lo + 1}-{hi}")

        indications = {i for i in extract_indications(pdf, lo, hi) if len(i) > 3}
        exclude = set(indications)
        for ind in indications:
            for _, name, pt in lut.get(ind, []):
                exclude.add(name.lower())
                if pt:
                    exclude.add(pt.lower())
            # "atopic dermatitis" being an indication also rules out the bare
            # "dermatitis" that appears inside it; likewise "sinusitis" inside
            # "chronic rhinosinusitis...". Those name the disease being
            # treated, not an adverse reaction. Substring rather than
            # word-split, so "sinusitis" in "rhinosinusitis" is caught too.
            exclude.update(w for w in ind.split() if len(w) >= 6)
            exclude.update(t for t in lut if len(t) >= 6 and t in ind)

        # Administrative MedDRA terms that pass the semantic-type filter but
        # are never a reported reaction in this context.
        exclude.update({"adverse event", "adverse reaction", "overdose", "death"})
        print(f"{len(indications)} indications excluded: {', '.join(sorted(indications))}")

        ae_pages = section_pages(pdf, lo, hi, r"4\.8\s+Undesirable effects")
        start = ae_pages[0] if ae_pages else lo
        end = next(iter(section_pages(pdf, start, hi, r"4\.9\s+Overdose")), hi)

        table_rows = []
        for i in range(start, end + 1):
            table_rows += extract_table(pdf.pages[i])
        narrative = "\n".join((pdf.pages[i].extract_text() or "") for i in range(start, end + 1))

    rows = []
    for r in table_rows:
        for t in decompose(r["term"]):
            pt, tty, note = resolve(t, lut)
            rows.append({
                "Source": "Table", "SOC": r["soc"], "Frequency": r["frequency"],
                "Term as written": t, "PT": pt or "", "Level": tty or "",
                "Finding": note, "Confidence": "high", "Context": "",
            })

    seen = {r["PT"] for r in rows if r["PT"]}

    for head in narrative_subheadings(narrative):
        # Subheadings also group by indication ("Paediatric population" ->
        # "Atopic dermatitis" -> ...), so the same exclusion applies here.
        if head.lower() in exclude:
            continue
        pt, tty, note = resolve(head, lut)
        if not pt or pt in seen or (pt.lower() in exclude):
            continue
        seen.add(pt)
        rows.append({
            "Source": "Narrative heading", "SOC": "", "Frequency": "",
            "Term as written": head, "PT": pt, "Level": tty or "",
            "Finding": note, "Confidence": "high", "Context": "",
        })

    marker = "Description of selected adverse reactions"
    prose = narrative[narrative.find(marker):] if marker in narrative else narrative
    for term, ctx in scan_narrative(prose, lut, exclude, ae_only).items():
        pt, tty, note = resolve(term, lut)
        if not pt or pt in seen:
            continue
        seen.add(pt)
        rows.append({
            "Source": "Narrative prose", "SOC": "", "Frequency": "",
            "Term as written": term, "PT": pt, "Level": tty or "",
            "Finding": note, "Confidence": "review", "Context": ctx.strip(),
        })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    findings = [r for r in rows if r["Finding"]]
    print(f"\n-> {out_path}")
    print(f"   {sum(r['Source'] == 'Table' for r in rows)} from the tabulated list, "
          f"{sum(r['Source'] == 'Narrative' for r in rows)} from the narrative")
    print(f"   {len(findings)} coding finding(s):")
    for r in findings:
        print(f"     · {r['Term as written']}: {r['Finding']}")


if __name__ == "__main__":
    main()
