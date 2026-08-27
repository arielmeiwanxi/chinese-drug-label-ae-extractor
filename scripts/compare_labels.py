"""Reconcile the adverse reactions of two labels at MedDRA Preferred Term level.

The two documents are not directly comparable as written. A local label lists
reactions at Lowest Level Term granularity and a reference document lists them
at Preferred Term granularity, so a term present in both can still look
different: "Corneal ulcer" and "Ulcerative keratitis" are the same concept.
Comparing the raw strings reports those as discrepancies; comparing after
roll-up does not. Every term on both sides is therefore resolved to its PT
before any set arithmetic happens.

Both inputs are CSVs. Column names are auto-detected so the reference side can
come from extract_smpc_ae.py and the local side from either the curated
reference extraction or a future extractor, without a schema change here.

Usage:
  python compare_labels.py <reference.csv> <local.csv> <meddra_index.parquet> <out.csv>
"""
import csv
import sys

import pandas as pd

# Candidate column names, in priority order, for the English term to resolve
# and for the original wording worth keeping for provenance.
TERM_COLS = ["PT", "Standard English Term", "Term as written", "term"]
SOURCE_COLS = ["Chinese Term", "Term as written", "Source Section", "Section"]


def pick(row, candidates):
    for c in candidates:
        if row.get(c):
            return row[c].strip()
    return ""


def load_lookup(index_path):
    idx = pd.read_parquet(index_path)
    lut = {}
    for r in idx.itertuples():
        lut.setdefault(r.term_lower, []).append((r.tty, r.term, r.pt_term))
    return lut


def to_pt(term, lut):
    """Resolve a term to its Preferred Term. Returns (pt, note)."""
    hits = lut.get(term.lower().strip())
    if not hits:
        return None, "not in MedDRA"
    pt_hit = next((h for h in hits if h[0] == "PT"), None)
    if pt_hit:
        return pt_hit[1], ""
    _, name, pt = hits[0]
    return pt, f"LLT '{name}' rolled up to PT '{pt}'"


def load_side(path, lut):
    """Map PT -> {as_written, note, source}. Terms that cannot be resolved are
    kept separately: dropping them silently would understate the differences."""
    resolved, unresolved, sources = {}, [], set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            term = pick(row, TERM_COLS)
            if not term:
                continue
            src = (row.get("Source") or "").strip()
            sources.add(src.split()[0] if src else "")
            pt, note = to_pt(term, lut)
            written = pick(row, SOURCE_COLS) or term
            if pt is None:
                unresolved.append({"as_written": written, "term": term})
            else:
                resolved.setdefault(pt, {"as_written": written, "note": note, "source": src})
    return resolved, unresolved, sources


def coverage_warning(ref_sources, loc_sources):
    """Comparing a fully extracted side against a partially extracted one
    manufactures differences. A term the other side never had a chance to
    report is not a discrepancy, so say so rather than letting the counts
    imply otherwise."""
    ref_labelled = {s for s in ref_sources if s}
    loc_labelled = {s for s in loc_sources if s}
    # An unlabelled side is not an uncovered side. A hand-curated list carries
    # no Source column but may well cover everything, so comparing coverage
    # against it would raise a warning that is simply wrong.
    if not ref_labelled or not loc_labelled:
        return []
    ref_only = ref_labelled - loc_labelled
    loc_only = loc_labelled - ref_labelled
    msgs = []
    if ref_only:
        msgs.append(f"reference covers {', '.join(sorted(ref_only))} sections that the local side does not")
    if loc_only:
        msgs.append(f"local covers {', '.join(sorted(loc_only))} sections that the reference does not")
    return msgs


def main():
    if len(sys.argv) != 5:
        print(__doc__.strip().splitlines()[-1])
        sys.exit(1)
    ref_path, loc_path, index_path, out_path = sys.argv[1:5]

    lut = load_lookup(index_path)
    ref, ref_bad, ref_src = load_side(ref_path, lut)
    loc, loc_bad, loc_src = load_side(loc_path, lut)
    warnings = coverage_warning(ref_src, loc_src)

    rows = []
    for pt in sorted(set(ref) | set(loc)):
        in_ref, in_loc = pt in ref, pt in loc
        rows.append({
            "PT": pt,
            "Status": "In both" if in_ref and in_loc else
                      ("Reference only" if in_ref else "Local only"),
            "Reference wording": ref.get(pt, {}).get("as_written", ""),
            "Local wording": loc.get(pt, {}).get("as_written", ""),
            "Note": ref.get(pt, {}).get("note", "") or loc.get(pt, {}).get("note", ""),
            "Found in": ref.get(pt, {}).get("source", "") or loc.get(pt, {}).get("source", ""),
        })
    for side, bad in (("Reference", ref_bad), ("Local", loc_bad)):
        for b in bad:
            rows.append({
                "PT": "", "Status": f"{side}: unresolved",
                "Reference wording": b["as_written"] if side == "Reference" else "",
                "Local wording": b["as_written"] if side == "Local" else "",
                "Note": f"'{b['term']}' not in MedDRA - needs manual coding",
                "Found in": "",
            })

    order = {"In both": 0, "Reference only": 1, "Local only": 2}
    rows.sort(key=lambda r: (order.get(r["Status"], 3), r["PT"]))
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {s: sum(r["Status"] == s for r in rows) for s in order}
    rolled = sum(1 for r in rows if r["Note"].startswith("LLT"))
    print(f"-> {out_path}")
    print(f"   in both: {counts['In both']}")
    print(f"   reference only: {counts['Reference only']}")
    print(f"   local only: {counts['Local only']}")
    print(f"   unresolved: {len(ref_bad) + len(loc_bad)}")
    print(f"   {rolled} term(s) matched only after LLT->PT roll-up "
          f"(these would be false discrepancies under exact matching)")
    for w in warnings:
        print(f"\n   WARNING: {w}.")
        print("            Differences arising from that gap are not real "
              "discrepancies; check the 'Found in' column before acting on them.")


if __name__ == "__main__":
    main()
